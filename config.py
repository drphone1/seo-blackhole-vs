# config.py
# Created: 2025-02-03 13:44:02 UTC
# Last modified: 2025-02-03 18:13:19 UTC
# Author: drphon
# Repository: drphon/chat-6-deepseek

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json

@dataclass
class ScrapingConfig:
    max_retries: int = 3
    timeout: int = 30
    rate_limit: float = 1.0
    max_concurrent: int = 5
    js_wait_time: float = 2.0
    max_url_length: int = 2000
    user_agent_rotate: bool = True
    verify_ssl: bool = True
    follow_redirects: bool = True
    max_redirects: int = 5
    respect_robots_txt: bool = True
    cookies_enabled: bool = True
    session_timeout: int = 3600
    max_response_size: int = 50 * 1024 * 1024  # 50MB
    retry_statuses: tuple = (500, 502, 503, 504)
    retry_delay: float = 1.0
    retry_backoff: float = 2.0

@dataclass
class ResourceConfig:
    max_memory_percent: float = 85.0
    max_cpu_percent: float = 90.0
    cleanup_threshold: float = 75.0
    memory_check_interval: int = 60
    process_monitor_enabled: bool = True
    thread_monitor_enabled: bool = True
    io_monitor_enabled: bool = True
    network_monitor_enabled: bool = True
    max_threads: int = 50
    thread_timeout: int = 300
    gc_threshold: tuple = (700, 10, 10)
    min_free_memory: int = 100 * 1024 * 1024  # 100MB

@dataclass
class DownloadConfig:
    max_concurrent: int = 3
    rate_limit: float = 2.0
    chunk_size: int = 8192
    cache_size: int = 100 * 1024 * 1024  # 100MB
    max_file_size: int = 1024 * 1024 * 1024  # 1GB
    supported_protocols: tuple = ('http://', 'https://')
    download_timeout: int = 600
    verify_hash: bool = True
    resume_support: bool = True
    auto_decompress: bool = True

@dataclass
class CacheConfig:
    enabled: bool = True
    max_size: int = 500 * 1024 * 1024  # 500MB
    expiration: int = 3600 * 24  # 24 hours
    cleanup_interval: int = 3600
    compression_enabled: bool = True
    compression_level: int = 6
    memory_cache_size: int = 50 * 1024 * 1024  # 50MB
    disk_cache_size: int = 450 * 1024 * 1024  # 450MB
    cache_dir: str = 'cache'

@dataclass
class LoggingConfig:
    level: str = 'INFO'
    format: str = '%(asctime)s UTC - %(name)s - %(levelname)s - %(message)s'
    file: str = 'scraper.log'
    max_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_output: bool = True
    file_output: bool = True
    syslog_output: bool = False
    log_requests: bool = True
    log_responses: bool = False
    log_errors: bool = True

@dataclass
class SecurityConfig:
    ssl_verify: bool = True
    ssl_cert_path: Optional[str] = None
    allowed_domains: tuple = field(default_factory=tuple)
    blocked_domains: tuple = field(default_factory=tuple)
    request_timeout: int = 30
    max_request_size: int = 50 * 1024 * 1024  # 50MB
    sanitize_headers: bool = True
    validate_urls: bool = True

@dataclass
class ErrorConfig:
    max_error_count: int = 1000
    error_threshold: float = 0.3
    retry_codes: tuple = (500, 502, 503, 504)
    ignore_errors: tuple = (404,)
    alert_on_error: bool = True
    error_cooldown: int = 300
    error_grouping: bool = True

@dataclass
class SystemConfig:
    base_dir: Path = Path(os.getcwd())
    temp_dir: Path = Path(os.getcwd()) / 'temp'
    output_dir: Path = Path(os.getcwd()) / 'output'
    working_dir: Path = Path(os.getcwd()) / 'working'
    debug_mode: bool = False
    testing_mode: bool = False
    development_mode: bool = False

class Configuration:
    def __init__(self):
        self.scraping = ScrapingConfig()
        self.resources = ResourceConfig()
        self.download = DownloadConfig()
        self.cache = CacheConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        self.error = ErrorConfig()
        self.system = SystemConfig()

    def load_from_file(self, path: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                self._update_from_dict(data)
        except Exception as e:
            print(f"Error loading config: {e}")

    def save_to_file(self, path: str) -> None:
        """Save current configuration to JSON file."""
        try:
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'scraping': self._dataclass_to_dict(self.scraping),
            'resources': self._dataclass_to_dict(self.resources),
            'download': self._dataclass_to_dict(self.download),
            'cache': self._dataclass_to_dict(self.cache),
            'logging': self._dataclass_to_dict(self.logging),
            'security': self._dataclass_to_dict(self.security),
            'error': self._dataclass_to_dict(self.error),
            'system': self._dataclass_to_dict(self.system)
        }

    def _dataclass_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert a dataclass instance to dictionary."""
        return {k: str(v) if isinstance(v, Path) else v 
                for k, v in obj.__dict__.items()}

    def _update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for section, values in data.items():
            if hasattr(self, section):
                config_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)

    def validate(self) -> bool:
        """Validate configuration values."""
        try:
            assert 0 < self.scraping.max_retries < 10
            assert 0 < self.scraping.timeout < 300
            assert 0 < self.resources.max_memory_percent <= 95
            assert 0 < self.download.max_concurrent < 10
            assert 0 < self.cache.max_size < 10 * 1024 * 1024 * 1024  # 10GB
            return True
        except AssertionError:
            return False

# Create global configuration instance
CONFIG = Configuration()

# Last modified: 2025-02-03 18:13:19 UTC
# End of config.py