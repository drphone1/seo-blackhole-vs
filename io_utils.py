# io_utils.py
# Created: 2025-02-03 13:44:02 UTC
# Last modified: 2025-02-03 18:18:04 UTC
# Author: drphon
# Repository: drphon/chat-6-deepseek

import os
import gzip
import hashlib
import aiohttp
import aiofiles
import logging
import asyncio
import tempfile
from typing import Dict, List, Optional, Union, BinaryIO, Any, Set
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from config import CONFIG
from exceptions import (
    FileSystemError,
    NetworkError,
    CacheError,
    ValidationError,
    error_tracker
)

logger = logging.getLogger(__name__)

@dataclass
class DownloadStats:
    """Statistics for download operations."""
    url: str
    size: int
    duration: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status_code: Optional[int] = None
    error: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)

class DownloadManager:
    """Advanced async file download management."""
    
    def __init__(
        self,
        max_concurrent: int = CONFIG.download.max_concurrent,
        rate_limit: float = CONFIG.download.rate_limit,
        chunk_size: int = CONFIG.download.chunk_size,
        cache_size: int = CONFIG.download.cache_size,
        verify_hash: bool = CONFIG.download.verify_hash,
        resume_support: bool = CONFIG.download.resume_support
    ):
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit
        self.chunk_size = chunk_size
        self.cache_size = cache_size
        self.verify_hash = verify_hash
        self.resume_support = resume_support
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._downloads: Dict[str, DownloadStats] = {}
        self._last_request = datetime.min
        self._cache: Dict[str, bytes] = {}
        self._cache_usage = 0
        self._lock = asyncio.Lock()
    
    async def __aenter__(self):
        """Initialize async session."""
        timeout = aiohttp.ClientTimeout(total=CONFIG.download.download_timeout)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup async session."""
        if self._session:
            await self._session.close()
    
    async def download(
        self,
        url: str,
        destination: Union[str, Path],
        headers: Optional[Dict[str, str]] = None,
        verify_ssl: bool = True
    ) -> DownloadStats:
        """Download file with advanced features and statistics."""
        start_time = datetime.utcnow()
        stats = DownloadStats(url=url, size=0, duration=0, success=False)
        
        try:
            # Rate limiting
            await self._apply_rate_limit()
            
            # Check cache
            if url in self._cache:
                return await self._handle_cached_download(url, destination, start_time)
            
            async with self._semaphore:
                stats = await self._perform_download(
                    url, destination, headers, verify_ssl, start_time
                )
            
            return stats
            
        except Exception as e:
            logger.error(f"Download failed for {url}: {str(e)}")
            stats.error = str(e)
            error_tracker.record_error(e)
            raise NetworkError(f"Download failed: {str(e)}", url=url)
        
        finally:
            self._downloads[url] = stats
    
    async def _perform_download(
        self,
        url: str,
        destination: Union[str, Path],
        headers: Optional[Dict[str, str]],
        verify_ssl: bool,
        start_time: datetime
    ) -> DownloadStats:
        """Perform actual download operation."""
        destination = Path(destination)
        temp_file = None
        file_hash = hashlib.sha256()
        
        try:
            headers = headers or {}
            if self.resume_support and destination.exists():
                headers['Range'] = f'bytes={destination.stat().st_size}-'
            
            async with self._session.get(
                url,
                headers=headers,
                ssl=verify_ssl
            ) as response:
                response.raise_for_status()
                stats = DownloadStats(
                    url=url,
                    size=0,
                    duration=0,
                    success=False,
                    status_code=response.status,
                    headers=dict(response.headers)
                )
                
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                
                async with aiofiles.open(temp_file.name, 'wb') as f:
                    async for chunk in response.content.iter_chunked(self.chunk_size):
                        await f.write(chunk)
                        if self.verify_hash:
                            file_hash.update(chunk)
                        stats.size += len(chunk)
                
                # Verify hash if provided
                if self.verify_hash and 'X-Content-Hash' in response.headers:
                    if file_hash.hexdigest() != response.headers['X-Content-Hash']:
                        raise ValidationError("File hash verification failed")
                
                # Move temporary file to destination
                os.replace(temp_file.name, destination)
                
                stats.duration = (datetime.utcnow() - start_time).total_seconds()
                stats.success = True
                
                # Cache small files
                if stats.size <= self.cache_size:
                    await self._cache_file(url, destination)
                
                return stats
                
        except Exception as e:
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise
    
    async def _handle_cached_download(
        self,
        url: str,
        destination: Union[str, Path],
        start_time: datetime
    ) -> DownloadStats:
        """Handle download from cache."""
        destination = Path(destination)
        async with aiofiles.open(destination, 'wb') as f:
            await f.write(self._cache[url])
        
        return DownloadStats(
            url=url,
            size=len(self._cache[url]),
            duration=(datetime.utcnow() - start_time).total_seconds(),
            success=True,
            status_code=200,
            headers={'X-Cache': 'HIT'}
        )
    
    async def _cache_file(self, url: str, path: Path) -> None:
        """Cache file content in memory."""
        async with self._lock:
            if self._cache_usage + path.stat().st_size > self.cache_size:
                self._clear_cache()
            
            async with aiofiles.open(path, 'rb') as f:
                content = await f.read()
                self._cache[url] = content
                self._cache_usage += len(content)
    
    def _clear_cache(self) -> None:
        """Clear download cache."""
        self._cache.clear()
        self._cache_usage = 0
    
    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting to downloads."""
        async with self._lock:
            now = datetime.utcnow()
            if (now - self._last_request).total_seconds() < self.rate_limit:
                await asyncio.sleep(self.rate_limit)
            self._last_request = now
    
    def get_stats(self) -> Dict[str, Any]:
        """Get download statistics."""
        return {
            'total_downloads': len(self._downloads),
            'successful_downloads': sum(1 for s in self._downloads.values() if s.success),
            'failed_downloads': sum(1 for s in self._downloads.values() if not s.success),
            'total_bytes': sum(s.size for s in self._downloads.values()),
            'cache_size': self._cache_usage,
            'cache_items': len(self._cache)
        }

class FileCache:
    """Advanced file-based cache system with compression."""
    
    def __init__(
        self,
        cache_dir: str,
        max_size: int = CONFIG.cache.max_size,
        expiration: int = CONFIG.cache.expiration,
        compression: bool = CONFIG.cache.compression_enabled,
        compression_level: int = CONFIG.cache.compression_level
    ):
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.expiration = expiration
        self.compression = compression
        self.compression_level = compression_level
        
        self._init_cache_dir()
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        
    def _init_cache_dir(self) -> None:
        """Initialize cache directory."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Load existing cache metadata
            self._load_metadata()
        except Exception as e:
            logger.error(f"Cache initialization failed: {str(e)}")
            raise CacheError("Failed to initialize cache", original_error=e)
    
    def _load_metadata(self) -> None:
        """Load cache metadata from disk."""
        metadata_file = self.cache_dir / 'metadata.json'
        if metadata_file.exists():
            try:
                async with aiofiles.open(metadata_file, 'r') as f:
                    content = await f.read()
                    self._metadata = json.loads(content)
            except Exception as e:
                logger.error(f"Failed to load cache metadata: {str(e)}")
                self._metadata = {}
    
    async def get(self, key: str) -> Optional[bytes]:
        """Get cached content with metadata validation."""
        try:
            path = self.cache_dir / self._hash_key(key)
            metadata = self._metadata.get(key, {})
            
            if not path.exists():
                return None
            
            # Check expiration
            if metadata.get('expires_at'):
                expires_at = datetime.fromisoformat(metadata['expires_at'])
                if datetime.utcnow() > expires_at:
                    await self._remove_item(key)
                    return None
            
            async with aiofiles.open(path, 'rb') as f:
                content = await f.read()
                
                # Decompress if needed
                if metadata.get('compressed', False):
                    content = gzip.decompress(content)
                
                # Verify hash
                if metadata.get('hash'):
                    current_hash = hashlib.sha256(content).hexdigest()
                    if current_hash != metadata['hash']:
                        logger.warning(f"Cache corruption detected for {key}")
                        await self._remove_item(key)
                        return None
                
                return content
                
        except Exception as e:
            logger.error(f"Cache read error: {str(e)}")
            return None
    
    async def set(
        self,
        key: str,
        content: bytes,
        expire_in: Optional[int] = None
    ) -> bool:
        """Set cache content with metadata."""
        async with self._lock:
            try:
                # Check cache size
                if self._get_cache_size() + len(content) > self.max_size:
                    await self._cleanup()
                
                path = self.cache_dir / self._hash_key(key)
                
                # Prepare content
                if self.compression:
                    content = gzip.compress(content, self.compression_level)
                
                # Update metadata
                self._metadata[key] = {
                    'size': len(content),
                    'created_at': datetime.utcnow().isoformat(),
                    'expires_at': (
                        datetime.utcnow() + timedelta(seconds=expire_in or self.expiration)
                    ).isoformat(),
                    'hash': hashlib.sha256(content).hexdigest(),
                    'compressed': self.compression
                }
                
                # Write content
                async with aiofiles.open(path, 'wb') as f:
                    await f.write(content)
                
                # Save metadata
                await self._save_metadata()
                return True
                
            except Exception as e:
                logger.error(f"Cache write error: {str(e)}")
                return False
    
    async def _cleanup(self) -> None:
        """Clean up expired and excess cache items."""
        now = datetime.utcnow()
        items = []
        
        for key, metadata in self._metadata.items():
            expires_at = datetime.fromisoformat(metadata['expires_at'])
            items.append((key, expires_at, metadata['size']))
        
        # Sort by expiration date
        items.sort(key=lambda x: x[1])
        
        # Remove expired and excess items
        current_size = self._get_cache_size()
        for key, expires_at, size in items:
            if expires_at < now or current_size > self.max_size:
                await self._remove_item(key)
                current_size -= size
    
    async def _remove_item(self, key: str) -> None:
        """Remove cache item and its metadata."""
        try:
            path = self.cache_dir / self._hash_key(key)
            if path.exists():
                path.unlink()
            if key in self._metadata:
                del self._metadata[key]
        except Exception as e:
            logger.error(f"Failed to remove cache item {key}: {str(e)}")
    
    def _get_cache_size(self) -> int:
        """Get current cache size."""
        return sum(metadata['size'] for metadata in self._metadata.values())
    
    async def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        metadata_file = self.cache_dir / 'metadata.json'
        async with aiofiles.open(metadata_file, 'w') as f:
            await f.write(json.dumps(self._metadata, indent=2))
    
    def _hash_key(self, key: str) -> str:
        """Create safe filename from key."""
        return f"{hashlib.sha256(key.encode()).hexdigest()}.cache"
    
    async def close(self) -> None:
        """Cleanup resources."""
        try:
            await self._save_metadata()
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {str(e)}")

# Create global instances
download_manager = DownloadManager()
file_cache = FileCache(CONFIG.cache.cache_dir)

# Cleanup handler
async def cleanup():
    """Cleanup IO resources."""
    await download_manager.__aexit__(None, None, None)
    await file_cache.close()

# Register cleanup handler
import atexit
atexit.register(lambda: asyncio.run(cleanup()))

# Last modified: 2025-02-03 18:18:04 UTC
# End of io_utils.py