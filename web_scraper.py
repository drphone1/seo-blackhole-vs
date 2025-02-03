# web_scraper.py
# Created: 2025-02-03 17:54:23 UTC
# Author: drphon
# Repository: drphon/chat-6-deepseek

import os
import re
import json
import time
import random
import logging
import asyncio
import aiohttp
import aiofiles
from typing import Dict, List, Optional, Union, Any, Set, Tuple, Type
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from fake_useragent import UserAgent

from config import CONFIG
from exceptions import (
    ScrapingError,
    NetworkError,
    ValidationError,
    RateLimitError,
    ProxyError,
    error_tracker,
    error_handler,
    ErrorCategory
)
from system_utils import MemoryManager, ProcessManager
from io_utils import DownloadManager, FileCache

# Configure logging with rotation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s UTC - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.RotatingFileHandler(
            'web_scraper.log',
            maxBytes=10485760,  # 10MB
            backupCount=5,
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ScrapedContent:
    """Data class for scraped content with enhanced validation and metadata.
    
    Attributes:
        url (str): Source URL of the content
        html (str): Raw HTML content
        title (str): Page title
        metadata (Dict[str, Any]): Extended metadata about the content
        timestamp (str): ISO formatted timestamp
        status_code (int): HTTP status code
        headers (Dict[str, str]): Response headers
        encoding (str): Content encoding
        size_bytes (int): Content size in bytes
        load_time (float): Time taken to load the content
        is_cached (bool): Whether content was served from cache
    """
    url: str
    html: str
    title: str
    metadata: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status_code: int = 200
    headers: Dict[str, str] = field(default_factory=dict)
    encoding: str = 'utf-8'
    size_bytes: int = field(init=False)
    load_time: float = 0.0
    is_cached: bool = False
    
    def __post_init__(self):
        """Validate scraped data and compute derived fields."""
        # Validate URL
        if not self.url or not urlparse(self.url).scheme:
            raise ValidationError("Invalid URL format", field="url", value=self.url)
        
        # Validate HTML content
        if not self.html.strip():
            raise ValidationError("Empty HTML content", field="html")
        
        # Validate status code
        if self.status_code < 200 or self.status_code >= 400:
            raise ValidationError(
                f"Invalid status code: {self.status_code}",
                field="status_code",
                value=self.status_code
            )
        
        # Compute content size
        self.size_bytes = len(self.html.encode(self.encoding))
        
        # Validate metadata
        if not isinstance(self.metadata, dict):
            raise ValidationError("Metadata must be a dictionary", field="metadata")
            
        # Add timestamp if not present
        self.metadata.setdefault('scraped_at', self.timestamp)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert scraped content to dictionary format."""
        return {
            'url': self.url,
            'title': self.title,
            'status_code': self.status_code,
            'headers': self.headers,
            'encoding': self.encoding,
            'size_bytes': self.size_bytes,
            'load_time': self.load_time,
            'is_cached': self.is_cached,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }
        
    async def save(self, path: Union[str, Path]) -> None:
        """Save scraped content to file."""
        path = Path(path)
        async with aiofiles.open(path, 'w', encoding=self.encoding) as f:
            await f.write(json.dumps(self.to_dict(), indent=2))

@dataclass
class ProxyConfig:
    """Configuration for proxy management."""
    url: str
    protocol: str
    username: Optional[str] = None
    password: Optional[str] = None
    last_used: datetime = field(default_factory=datetime.utcnow)
    fails: int = 0
    avg_response_time: float = 0.0
    
    def to_url(self) -> str:
        """Convert proxy configuration to URL format."""
        if self.username and self.password:
            return f"{self.protocol}://{self.username}:{self.password}@{self.url}"
        return f"{self.protocol}://{self.url}"
    
    def update_stats(self, response_time: float, failed: bool = False) -> None:
        """Update proxy statistics."""
        self.last_used = datetime.utcnow()
        if failed:
            self.fails += 1
        else:
            self.avg_response_time = (self.avg_response_time + response_time) / 2

@dataclass
class ScrapingStats:
    """Enhanced scraping statistics tracking."""
    start_time: datetime = field(default_factory=datetime.utcnow)
    urls_processed: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cached_requests: int = 0
    bytes_downloaded: int = 0
    total_load_time: float = 0.0
    js_renders: int = 0
    proxy_errors: int = 0
    rate_limits_hit: int = 0
    retries: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def average_load_time(self) -> float:
        """Calculate average load time per request."""
        total_requests = self.successful_requests + self.failed_requests
        return self.total_load_time / total_requests if total_requests > 0 else 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        total_requests = self.successful_requests + self.failed_requests
        return (self.successful_requests / total_requests * 100) if total_requests > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary format."""
        duration = datetime.utcnow() - self.start_time
        return {
            'duration_seconds': duration.total_seconds(),
            'urls_processed': self.urls_processed,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'cached_requests': self.cached_requests,
            'bytes_downloaded': self.bytes_downloaded,
            'average_load_time': self.average_load_time,
            'success_rate': self.success_rate,
            'js_renders': self.js_renders,
            'proxy_errors': self.proxy_errors,
            'rate_limits_hit': self.rate_limits_hit,
            'retries': self.retries,
            'errors': self.errors
        }
class WebScraper:
    """Enhanced web scraper with advanced features and protections.
    
    Features:
    - Async scraping capabilities
    - Advanced rate limiting
    - User agent rotation
    - JavaScript rendering
    - Smart retries
    - Cookie management
    - Request queueing
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        max_retries: int = CONFIG['scraping']['max_retries'],
        timeout: int = CONFIG['scraping']['timeout'],
        respect_robots: bool = True,
        js_rendering: bool = True,
        cache_enabled: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.max_retries = max_retries
        self.timeout = timeout
        self.respect_robots = respect_robots
        self.js_rendering = js_rendering
        
        # Initialize components
        self._setup_directories()
        self._setup_selenium() if js_rendering else None
        self._setup_memory_manager()
        self._setup_download_manager()
        
        # Initialize caching
        self.cache = FileCache('scraper_cache') if cache_enabled else None
        
        # Statistics tracking
        self.stats = ScrapingStats()
        
        # Request management
        self.user_agent = UserAgent()
        self._robots_cache: Dict[str, Set[str]] = {}
        self._rate_limiters: Dict[str, datetime] = {}
        self._cookie_jar: Dict[str, Dict[str, str]] = {}
        
        # Thread safety
        self._lock = asyncio.Lock()
        self._request_queue = asyncio.Queue()
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Initialize async resources."""
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup async resources."""
        await self.cleanup()

    def _setup_directories(self) -> None:
        """Setup required directories with proper permissions."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir = self.output_dir / 'cache'
            self.cache_dir.mkdir(exist_ok=True)
            
            logger.info("Directory structure initialized successfully")
        except Exception as e:
            logger.error(f"Directory setup failed: {str(e)}")
            raise ScrapingError("Failed to setup directories", original_error=e)

    def _setup_selenium(self) -> None:
        """Initialize Selenium with optimized settings."""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-logging')
            chrome_options.add_argument('--log-level=3')
            chrome_options.add_argument(f'--user-agent={self.user_agent.random}')
            
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(self.timeout)
            logger.info("Selenium initialized successfully")
        except Exception as e:
            logger.error(f"Selenium initialization failed: {str(e)}")
            raise ScrapingError("Failed to initialize Selenium", original_error=e)

    def _setup_memory_manager(self) -> None:
        """Initialize memory management."""
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

    @error_handler(error_category=ErrorCategory.SYSTEM)
    async def scrape(self, url: str) -> ScrapedContent:
        """Scrape a single URL with full protection and validation."""
        start_time = time.time()
        
        try:
            # Validate URL
            if not self._is_valid_url(url):
                raise ValidationError(f"Invalid URL format: {url}")
            
            # Check robots.txt
            if self.respect_robots and not await self._is_allowed_by_robots(url):
                raise ValidationError(f"URL not allowed by robots.txt: {url}")
            
            # Apply rate limiting
            await self._apply_rate_limiting(url)
            
            # Check cache first
            if self.cache:
                cached_content = await self.cache.get(url)
                if cached_content:
                    self.stats.cached_requests += 1
                    return self._create_scraped_content(
                        url, cached_content.decode(),
                        is_cached=True,
                        load_time=time.time() - start_time
                    )
            
            # Get content
            content = await self._get_content(url)
            
            # Update statistics
            load_time = time.time() - start_time
            self.stats.total_load_time += load_time
            self.stats.successful_requests += 1
            self.stats.bytes_downloaded += len(content)
            
            # Create and return scraped content
            scraped = self._create_scraped_content(url, content, load_time=load_time)
            
            # Cache the result
            if self.cache:
                await self.cache.set(url, content.encode())
            
            return scraped
            
        except Exception as e:
            self.stats.failed_requests += 1
            self.stats.errors.append({
                'url': url,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
            raise

    async def scrape_batch(
        self,
        urls: List[str],
        concurrency: Optional[int] = None
    ) -> List[ScrapedContent]:
        """Scrape multiple URLs in parallel with smart queuing."""
        if not urls:
            return []
        
        concurrency = concurrency or CONFIG['scraping']['max_concurrent']
        
        # Initialize queue
        queue = asyncio.Queue()
        for url in urls:
            await queue.put(url)
        
        # Process queue with concurrency control
        tasks = []
        results = []
        
        async def worker():
            while True:
                try:
                    url = await queue.get()
                    try:
                        result = await self.scrape(url)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error scraping {url}: {str(e)}")
                    finally:
                        queue.task_done()
                except asyncio.CancelledError:
                    break
        
        # Start workers
        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        
        # Wait for queue to be processed
        await queue.join()
        
        # Cancel workers
        for w in workers:
            w.cancel()
        
        # Wait for workers to close
        await asyncio.gather(*workers, return_exceptions=True)
        
        return results

    async def _get_content(self, url: str) -> str:
        """Get content with or without JavaScript rendering."""
        if self.js_rendering:
            content = await self._get_content_with_js(url)
            self.stats.js_renders += 1
        else:
            content = await self._get_content_simple(url)
        
        if not content:
            raise ScrapingError(f"Failed to get content from {url}")
        
        return content

    async def _get_content_with_js(self, url: str) -> str:
        """Get content with JavaScript rendering."""
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, self.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Wait for dynamic content
            time.sleep(CONFIG['scraping']['js_wait_time'])
            
            return self.driver.page_source
            
        except Exception as e:
            logger.error(f"JavaScript rendering failed: {str(e)}")
            raise ScrapingError("Failed to render JavaScript", original_error=e)

    async def _get_content_simple(self, url: str) -> str:
        """Get content without JavaScript rendering."""
        headers = {
            'User-Agent': self.user_agent.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        try:
            async with self._session.get(url, headers=headers) as response:
                if response.status == 429:
                    self.stats.rate_limits_hit += 1
                    raise RateLimitError(f"Rate limited by {url}")
                
                response.raise_for_status()
                return await response.text()
                
        except aiohttp.ClientError as e:
            raise NetworkError(f"Request failed for {url}", original_error=e)

    async def _is_allowed_by_robots(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        
        if domain not in self._robots_cache:
            robots_url = f"{domain}/robots.txt"
            try:
                async with self._session.get(robots_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        self._parse_robots_txt(domain, content)
                    else:
                        self._robots_cache[domain] = set()
            except Exception:
                self._robots_cache[domain] = set()
        
        return parsed.path not in self._robots_cache[domain]

    def _parse_robots_txt(self, domain: str, content: str) -> None:
        """Parse robots.txt content."""
        disallowed = set()
        for line in content.split('\n'):
            if line.startswith('Disallow:'):
                path = line.split(':', 1)[1].strip()
                if path:
                    disallowed.add(path)
        self._robots_cache[domain] = disallowed

    async def _apply_rate_limiting(self, url: str) -> None:
        """Apply rate limiting per domain."""
        parsed = urlparse(url)
        domain = parsed.netloc
        
        async with self._lock:
            last_request = self._rate_limiters.get(domain)
            if last_request:
                elapsed = (datetime.utcnow() - last_request).total_seconds()
                if elapsed < CONFIG['scraping']['rate_limit']:
                    wait_time = CONFIG['scraping']['rate_limit'] - elapsed
                    await asyncio.sleep(wait_time)
            
            self._rate_limiters[domain] = datetime.utcnow()

    def _create_scraped_content(
        self,
        url: str,
        html: str,
        is_cached: bool = False,
        load_time: float = 0.0
    ) -> ScrapedContent:
        """Create ScrapedContent object with metadata."""
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string if soup.title else ''
        
        metadata = {
            'length': len(html),
            'links': [a.get('href') for a in soup.find_all('a', href=True)],
            'images': len(soup.find_all('img')),
            'scripts': len(soup.find_all('script')),
            'meta_tags': {
                meta.get('name', meta.get('property', '')): meta.get('content', '')
                for meta in soup.find_all('meta')
                if meta.get('name') or meta.get('property')
            }
        }
        
        return ScrapedContent(
            url=url,
            html=html,
            title=title.strip() if title else '',
            metadata=metadata,
            is_cached=is_cached,
            load_time=load_time
        )

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format and allowed protocols."""
        try:
            result = urlparse(url)
            return all([
                result.scheme in ('http', 'https'),
                result.netloc,
                len(url) < CONFIG['scraping']['max_url_length']
            ])
        except Exception:
            return False

    async def cleanup(self) -> None:
        """Cleanup resources and generate final report."""
        try:
            if hasattr(self, 'driver'):
                self.driver.quit()
            
            if self._session:
                await self._session.close()
            
            if self.cache:
                await self.cache.close()
            
            # Generate final report
            await self._generate_report()
            
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")

    async def _generate_report(self) -> None:
        """Generate detailed scraping report."""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'stats': self.stats.to_dict(),
            'config': {
                'max_retries': self.max_retries,
                'timeout': self.timeout,
                'respect_robots': self.respect_robots,
                'js_rendering': self.js_rendering,
                'cache_enabled': bool(self.cache)
            }
        }
        
        report_file = self.output_dir / f"scraping_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        async with aiofiles.open(report_file, 'w') as f:
            await f.write(json.dumps(report, indent=2))
        
        logger.info(f"Scraping report generated: {report_file}")

# Last modified: 2025-02-03 18:00:40 UTC
# Modified by: drphon
# End of web_scraper.py