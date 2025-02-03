# content_processor.py
# Created: 2025-02-03 12:06:22 UTC
# Last modified: 2025-02-03 12:06:22 UTC
# Author: drphon
# Repository: drphon/chat-6-deepseek

import os
import re
import json
import time
import logging
import hashlib
import asyncio
import aiofiles
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

from config import CONFIG
from exceptions import (
    ProcessingError,
    ValidationError,
    DataConsistencyError,
    ResourceLimitError
)
from utils import MemoryManager, FileSystemWatcher

# Configure logging with rotation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s UTC - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.RotatingFileHandler(
            'processor.log',
            maxBytes=10485760,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessedContent:
    """Data class for processed content with validation."""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    timestamp: str
    hash: str
    
    def __post_init__(self):
        """Validate data after initialization."""
        if not self.url or not urlparse(self.url).scheme:
            raise ValidationError("Invalid URL format")
        if not self.title.strip():
            raise ValidationError("Empty title")
        if not self.content.strip():
            raise ValidationError("Empty content")
        if not isinstance(self.metadata, dict):
            raise ValidationError("Metadata must be a dictionary")

class ContentProcessor:
    """Enhanced content processor with advanced features and optimizations.
    
    Features:
    - Async processing capabilities
    - Smart content analysis
    - Advanced data validation
    - Memory optimization
    - Intelligent caching
    - Detailed reporting
    """
    
    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        batch_size: int = CONFIG['processing']['batch_size'],
        max_workers: int = CONFIG['processing']['max_workers'],
        cache_enabled: bool = True
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.cache_enabled = cache_enabled
        
        # Initialize components
        self._setup_directories()
        self._setup_memory_manager()
        self._setup_file_watcher()
        self._initialize_stats()
        
        # Thread safety
        self._lock = asyncio.Lock()
        self._process_queue = asyncio.Queue()
        self._processed_urls: Set[str] = set()
        
    def _setup_directories(self) -> None:
        """Setup required directories with proper permissions."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir = self.output_dir / 'cache'
            self.cache_dir.mkdir(exist_ok=True)
            
            # Setup subdirectories for different output types
            for subdir in ['json', 'excel', 'text', 'metadata']:
                (self.output_dir / subdir).mkdir(exist_ok=True)
                
            logger.info("Directory structure initialized successfully")
        except Exception as e:
            logger.error(f"Directory setup failed: {str(e)}")
            raise ProcessingError("Failed to setup directories", original_error=e)

    def _setup_memory_manager(self) -> None:
        """Initialize memory management with monitoring."""
        self.memory_manager = MemoryManager(
            threshold=CONFIG['resources']['max_memory_percent'],
            leak_detection=True
        )

    def _setup_file_watcher(self) -> None:
        """Initialize file system monitoring."""
        self.file_watcher = FileSystemWatcher(
            str(self.output_dir),
            patterns=["*.json", "*.xlsx", "*.txt"],
            recursive=True
        )

    def _initialize_stats(self) -> None:
        """Initialize processing statistics."""
        self.stats = {
            'start_time': datetime.utcnow(),
            'files_processed': 0,
            'successful_processing': 0,
            'failed_processing': 0,
            'total_bytes_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': []
        }

    async def process_content(self, content: Dict[str, Any]) -> ProcessedContent:
        """Process content with advanced validation and enrichment."""
        try:
            # Check cache first
            if self.cache_enabled:
                cached_result = await self._get_cached_result(content['url'])
                if cached_result:
                    async with self._lock:
                        self.stats['cache_hits'] += 1
                    return cached_result

            async with self._lock:
                self.stats['cache_misses'] += 1

            # Basic validation
            self._validate_input(content)
            
            # Process content
            processed = await self._process_single_content(content)
            
            # Post-processing validation
            self._validate_output(processed)
            
            # Update statistics
            async with self._lock:
                self.stats['successful_processing'] += 1
                self.stats['total_bytes_processed'] += len(processed.content)
                self._processed_urls.add(processed.url)
            
            # Cache result
            if self.cache_enabled:
                await self._cache_result(processed)
            
            return processed
            
        except Exception as e:
            async with self._lock:
                self.stats['failed_processing'] += 1
                self.stats['errors'].append(str(e))
            raise ProcessingError(f"Content processing failed for {content.get('url', 'unknown')}", original_error=e)

    def _validate_input(self, content: Dict[str, Any]) -> None:
        """Validate input content structure and data types."""
        required_fields = {'url', 'title', 'content'}
        if not all(field in content for field in required_fields):
            raise ValidationError(f"Missing required fields: {required_fields - set(content.keys())}")
        
        if not isinstance(content['content'], str):
            raise ValidationError("Content must be string type")

    def _validate_output(self, processed: ProcessedContent) -> None:
        """Validate processed content for consistency and completeness."""
        if not processed.content:
            raise ValidationError("Processed content is empty")
        
        if len(processed.content) < CONFIG['processing']['min_content_length']:
            raise ValidationError("Processed content too short")

    async def _process_single_content(self, content: Dict[str, Any]) -> ProcessedContent:
        """Process single content item with enrichment."""
        # Clean and normalize content
        cleaned_content = self._clean_content(content['content'])
        
        # Extract metadata
        metadata = await self._extract_metadata(cleaned_content)
        
        # Generate content hash
        content_hash = hashlib.sha256(cleaned_content.encode()).hexdigest()
        
        # Create processed content object
        processed = ProcessedContent(
            url=content['url'],
            title=content['title'],
            content=cleaned_content,
            metadata=metadata,
            timestamp=datetime.utcnow().isoformat(),
            hash=content_hash
        )
        
        return processed

    async def process_batch(self, contents: List[Dict[str, Any]]) -> List[ProcessedContent]:
        """Process multiple content items in parallel."""
        if not contents:
            return []
            
        tasks = []
        for content in contents:
            if content['url'] not in self._processed_urls:
                tasks.append(self.process_content(content))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing error: {str(result)}")
                continue
            processed_results.append(result)
        
        return processed_results

    def _clean_content(self, content: str) -> str:
        """Clean and normalize content."""
        # Remove HTML if present
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters
        text = re.sub(r'[^\w\s\.,!?-]', '', text)
        
        return text

    async def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from content."""
        return {
            'length': len(content),
            'word_count': len(content.split()),
            'sentence_count': len(re.split(r'[.!?]+', content)),
            'language': self._detect_language(content),
            'timestamp': datetime.utcnow().isoformat()
        }

    def _detect_language(self, text: str) -> str:
        """Detect content language."""
        # Implement language detection logic here
        return "en"  # Placeholder

    async def _get_cached_result(self, url: str) -> Optional[ProcessedContent]:
        """Retrieve cached processing result."""
        if not self.cache_enabled:
            return None
            
        cache_key = hashlib.md5(url.encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                async with aiofiles.open(cache_file, 'r') as f:
                    data = json.loads(await f.read())
                return ProcessedContent(**data)
            except Exception as e:
                logger.warning(f"Cache read failed for {url}: {str(e)}")
                
        return None

    async def _cache_result(self, processed: ProcessedContent) -> None:
        """Cache processing result."""
        if not self.cache_enabled:
            return
            
        try:
            cache_key = hashlib.md5(processed.url.encode()).hexdigest()
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            async with aiofiles.open(cache_file, 'w') as f:
                await f.write(json.dumps(asdict(processed), indent=2))
        except Exception as e:
            logger.warning(f"Cache write failed for {processed.url}: {str(e)}")

    def generate_report(self) -> Dict[str, Any]:
        """Generate detailed processing report."""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'duration': str(datetime.utcnow() - self.stats['start_time']),
            'stats': self.stats,
            'config': {
                'batch_size': self.batch_size,
                'max_workers': self.max_workers,
                'cache_enabled': self.cache_enabled
            }
        }
        
        report_file = self.output_dir / f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Processing report generated: {report_file}")
        return report

# Last modified: 2025-02-03 12:06:22 UTC
# End of content_processor.py