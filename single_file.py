# single_file.py
# Created: 2025-02-03 13:40:10 UTC
# Last modified: 2025-02-03 13:40:10 UTC
# Author: drphon
# Repository: drphon/chat-6-deepseek

import os
import re
import base64
import hashlib
import logging
import asyncio
import aiohttp
import aiofiles
from typing import Dict, List, Optional, Union, Any, Set
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from bs4 import BeautifulSoup
import magic  # for MIME type detection
from PIL import Image
from io import BytesIO

from config import CONFIG
from exceptions import (
    FileProcessingError,
    ResourceError,
    ValidationError,
    NetworkError
)
from utils import MemoryManager, DownloadManager

# Configure logging with rotation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s UTC - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.RotatingFileHandler(
            'single_file.log',
            maxBytes=10485760,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ResourceInfo:
    """Data class for resource information with validation."""
    url: str
    mime_type: str
    content: bytes
    hash: str
    size: int
    
    def __post_init__(self):
        """Validate resource data after initialization."""
        if not self.url or not urlparse(self.url).scheme:
            raise ValidationError("Invalid resource URL")
        if not self.mime_type:
            raise ValidationError("Missing MIME type")
        if not self.content:
            raise ValidationError("Empty content")
        if self.size <= 0:
            raise ValidationError("Invalid resource size")

class SingleFileProcessor:
    """Enhanced single file processor for converting web pages to self-contained HTML files.
    
    Features:
    - Async resource downloading
    - Smart resource optimization
    - Advanced caching
    - Resource validation
    - Memory optimization
    - Detailed reporting
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        max_size: int = CONFIG['resources']['max_file_size'],
        allowed_domains: Optional[List[str]] = None,
        cache_enabled: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.max_size = max_size
        self.allowed_domains = set(allowed_domains) if allowed_domains else set()
        self.cache_enabled = cache_enabled
        
        # Initialize components
        self._setup_directories()
        self._setup_memory_manager()
        self._setup_download_manager()
        self._initialize_stats()
        
        # Resource tracking
        self._processed_resources: Set[str] = set()
        self._resource_cache: Dict[str, ResourceInfo] = {}
        
        # Thread safety
        self._lock = asyncio.Lock()
        
    def _setup_directories(self) -> None:
        """Setup required directories with proper permissions."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir = self.output_dir / 'cache'
            self.cache_dir.mkdir(exist_ok=True)
            
            logger.info("Directory structure initialized successfully")
        except Exception as e:
            logger.error(f"Directory setup failed: {str(e)}")
            raise FileProcessingError("Failed to setup directories", original_error=e)

    def _setup_memory_manager(self) -> None:
        """Initialize memory management with monitoring."""
        self.memory_manager = MemoryManager(
            threshold=CONFIG['resources']['max_memory_percent'],
            leak_detection=True
        )

    def _setup_download_manager(self) -> None:
        """Initialize download management."""
        self.download_manager = DownloadManager(
            max_concurrent=CONFIG['download']['max_concurrent'],
            rate_limit=CONFIG['download']['rate_limit'],
            cache_size=CONFIG['download']['cache_size']
        )

    def _initialize_stats(self) -> None:
        """Initialize processing statistics."""
        self.stats = {
            'start_time': datetime.utcnow(),
            'files_processed': 0,
            'resources_processed': 0,
            'bytes_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': []
        }

    async def process_html(self, html: str, base_url: str) -> str:
        """Process HTML content and convert to self-contained file."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Process different resource types
            await asyncio.gather(
                self._process_styles(soup, base_url),
                self._process_scripts(soup, base_url),
                self._process_images(soup, base_url),
                self._process_fonts(soup, base_url)
            )
            
            # Add metadata
            self._add_metadata(soup, base_url)
            
            return str(soup)
            
        except Exception as e:
            logger.error(f"HTML processing failed: {str(e)}")
            raise FileProcessingError("Failed to process HTML", original_error=e)

    async def _process_styles(self, soup: BeautifulSoup, base_url: str) -> None:
        """Process and embed CSS styles."""
        for tag in soup.find_all(['link', 'style']):
            try:
                if tag.name == 'link' and tag.get('rel') == ['stylesheet']:
                    href = tag.get('href')
                    if href:
                        css_url = urljoin(base_url, href)
                        css_content = await self._get_resource(css_url, 'text/css')
                        if css_content:
                            new_style = soup.new_tag('style')
                            new_style.string = css_content.decode()
                            tag.replace_with(new_style)
            except Exception as e:
                logger.warning(f"Style processing error: {str(e)}")

    async def _process_scripts(self, soup: BeautifulSoup, base_url: str) -> None:
        """Process and embed JavaScript."""
        for script in soup.find_all('script', src=True):
            try:
                src = script.get('src')
                if src:
                    js_url = urljoin(base_url, src)
                    js_content = await self._get_resource(js_url, 'application/javascript')
                    if js_content:
                        script['src'] = None
                        script.string = js_content.decode()
            except Exception as e:
                logger.warning(f"Script processing error: {str(e)}")

    async def _process_images(self, soup: BeautifulSoup, base_url: str) -> None:
        """Process and embed images with optimization."""
        for img in soup.find_all('img', src=True):
            try:
                src = img.get('src')
                if src:
                    img_url = urljoin(base_url, src)
                    img_content = await self._get_resource(img_url, 'image/*')
                    if img_content:
                        optimized_content = self._optimize_image(img_content)
                        img['src'] = self._to_data_uri(optimized_content, 'image/jpeg')
            except Exception as e:
                logger.warning(f"Image processing error: {str(e)}")

    async def _process_fonts(self, soup: BeautifulSoup, base_url: str) -> None:
        """Process and embed web fonts."""
        for font_face in soup.find_all('style'):
            try:
                if '@font-face' in str(font_face):
                    # Extract and process font URLs
                    font_urls = re.findall(r'url\([\'"]?([^\'"]+)[\'"]?\)', str(font_face))
                    for url in font_urls:
                        font_url = urljoin(base_url, url)
                        font_content = await self._get_resource(font_url, 'font/*')
                        if font_content:
                            data_uri = self._to_data_uri(font_content, 'font/woff2')
                            font_face.string = font_face.string.replace(url, data_uri)
            except Exception as e:
                logger.warning(f"Font processing error: {str(e)}")

    async def _get_resource(self, url: str, mime_type: str) -> Optional[bytes]:
        """Get resource with caching and validation."""
        try:
            # Check cache first
            resource_hash = hashlib.md5(url.encode()).hexdigest()
            
            if resource_hash in self._resource_cache:
                self.stats['cache_hits'] += 1
                return self._resource_cache[resource_hash].content
                
            self.stats['cache_misses'] += 1
            
            # Validate URL
            if not self._is_allowed_url(url):
                logger.warning(f"URL not allowed: {url}")
                return None
                
            # Download resource
            async with self.download_manager as dm:
                content = await dm.download(url)
                
            # Validate content
            if not self._validate_resource(content, mime_type):
                return None
                
            # Cache resource
            resource_info = ResourceInfo(
                url=url,
                mime_type=mime_type,
                content=content,
                hash=resource_hash,
                size=len(content)
            )
            self._resource_cache[resource_hash] = resource_info
            
            return content
            
        except Exception as e:
            logger.error(f"Resource download failed: {str(e)}")
            return None

    def _is_allowed_url(self, url: str) -> bool:
        """Check if URL is allowed based on domain restrictions."""
        if not self.allowed_domains:
            return True
            
        try:
            domain = urlparse(url).netloc
            return domain in self.allowed_domains
        except Exception:
            return False

    def _validate_resource(self, content: bytes, expected_mime: str) -> bool:
        """Validate resource content and type."""
        if not content:
            return False
            
        try:
            actual_mime = magic.from_buffer(content, mime=True)
            if expected_mime.endswith('/*'):
                return actual_mime.split('/')[0] == expected_mime.split('/')[0]
            return actual_mime == expected_mime
        except Exception:
            return False

    def _optimize_image(self, content: bytes) -> bytes:
        """Optimize image for web embedding."""
        try:
            img = Image.open(BytesIO(content))
            
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Resize if too large
            max_dimension = CONFIG['resources']['max_image_dimension']
            if max(img.size) > max_dimension:
                ratio = max_dimension / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.LANCZOS)
            
            # Optimize
            output = BytesIO()
            img.save(output, format='JPEG', quality=85, optimize=True)
            return output.getvalue()
            
        except Exception as e:
            logger.warning(f"Image optimization failed: {str(e)}")
            return content

    def _to_data_uri(self, content: bytes, mime_type: str) -> str:
        """Convert content to data URI."""
        b64 = base64.b64encode(content).decode()
        return f"data:{mime_type};base64,{b64}"

    def _add_metadata(self, soup: BeautifulSoup, base_url: str) -> None:
        """Add processing metadata to the document."""
        meta = soup.new_tag('meta')
        meta['name'] = 'single-file-processor'
        meta['content'] = json.dumps({
            'processed_at': datetime.utcnow().isoformat(),
            'base_url': base_url,
            'processor_version': CONFIG['version']
        })
        soup.head.append(meta)

    def generate_report(self) -> Dict[str, Any]:
        """Generate detailed processing report."""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'duration': str(datetime.utcnow() - self.stats['start_time']),
            'stats': self.stats,
            'config': {
                'max_size': self.max_size,
                'allowed_domains': list(self.allowed_domains),
                'cache_enabled': self.cache_enabled
            }
        }
        
        report_file = self.output_dir / f"single_file_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Processing report generated: {report_file}")
        return report

# Last modified: 2025-02-03 13:40:10 UTC
# End of single_file.py