cache_key (str): Cache key involved
        operation (str): Cache operation that failed
        cache_type (str): Type of cache (memory, disk, etc.)
        key_pattern (str): Pattern of the cache key
        cache_stats (Dict[str, Any]): Cache statistics
    """
    def __init__(
        self,
        message: str,
        cache_key: str,
        operation: str,
        cache_type: str = "disk",
        key_pattern: Optional[str] = None,
        cache_stats: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
        retryable: bool = True,
        log_error: bool = True
    ) -> None:
        details = {
            "cache_key": cache_key,
            "operation": operation,
            "cache_type": cache_type,
            "key_pattern": key_pattern,
            "cache_stats": cache_stats or {},
            "error_type": type(original_error).__name__ if original_error else None,
            "error_message": str(original_error) if original_error else None
        }
        
        super().__init__(
            message,
            code="CACHE_ERROR",
            details=details,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.DATA,
            source="CacheManager",
            retryable=retryable,
            log_error=log_error,
            tags=["cache", cache_type, operation.lower()]
        )

class FileSystemError(BaseError):
    """
    File system operation related errors.
    
    Additional Attributes:
        path (Union[str, Path]): Path to the file or directory
        operation (str): File system operation that failed
        permissions (Optional[int]): File permissions
        file_stats (Dict[str, Any]): File statistics
        is_directory (bool): Whether the path is a directory
    """
    def __init__(
        self,
        message: str,
        path: Union[str, Path],
        operation: str,
        permissions: Optional[int] = None,
        file_stats: Optional[Dict[str, Any]] = None,
        is_directory: bool = False,
        original_error: Optional[Exception] = None,
        retryable: bool = True,
        log_error: bool = True
    ) -> None:
        path = Path(path)
        details = {
            "path": str(path),
            "operation": operation,
            "permissions": permissions,
            "file_stats": file_stats or {},
            "is_directory": is_directory,
            "parent_exists": path.parent.exists(),
            "path_exists": path.exists(),
            "error_type": type(original_error).__name__ if original_error else None,
            "error_message": str(original_error) if original_error else None
        }
        
        super().__init__(
            message,
            code="FILESYSTEM_ERROR",
            details=details,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.SYSTEM,
            source="FileSystem",
            retryable=retryable,
            log_error=log_error,
            tags=["filesystem", operation.lower(), "directory" if is_directory else "file"]
        )

class NetworkError(BaseError):
    """
    Network operation related errors.
    
    Additional Attributes:
        host (str): Target host
        port (Optional[int]): Target port
        protocol (str): Network protocol
        connection_info (Dict[str, Any]): Connection information
        network_stats (Dict[str, Any]): Network statistics
    """
    def __init__(
        self,
        message: str,
        host: str,
        port: Optional[int] = None,
        protocol: str = "TCP",
        connection_info: Optional[Dict[str, Any]] = None,
        network_stats: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
        retryable: bool = True,
        log_error: bool = True
    ) -> None:
        details = {
            "host": host,
            "port": port,
            "protocol": protocol,
            "connection_info": connection_info or {},
            "network_stats": network_stats or {},
            "error_type": type(original_error).__name__ if original_error else None,
            "error_message": str(original_error) if original_error else None
        }
        
        super().__init__(
            message,
            code="NETWORK_ERROR",
            details=details,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.NETWORK,
            source="NetworkManager",
            retryable=retryable,
            log_error=log_error,
            tags=["network", protocol.lower()]
        )

class SecurityError(BaseError):
    """
    Security-related errors.
    
    Additional Attributes:
        security_context (Dict[str, Any]): Security context information
        violation_type (str): Type of security violation
        affected_resource (str): Resource affected by the security violation
        security_level (str): Security level of the operation
    """
    def __init__(
        self,
        message: str,
        security_context: Optional[Dict[str, Any]] = None,
        violation_type: str = "UNKNOWN",
        affected_resource: str = "",
        security_level: str = "HIGH",
        original_error: Optional[Exception] = None,
        retryable: bool = False,
        log_error: bool = True
    ) -> None:
        details = {
            "security_context": security_context or {},
            "violation_type": violation_type,
            "affected_resource": affected_resource,
            "security_level": security_level,
            "error_type": type(original_error).__name__ if original_error else None,
            "error_message": str(original_error) if original_error else None
        }
        
        super().__init__(
            message,
            code="SECURITY_ERROR",
            details=details,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SECURITY,
            source="SecurityManager",
            retryable=retryable,
            log_error=log_error,
            tags=["security", violation_type.lower(), security_level.lower()]
        )

class ProcessError(BaseError):
    """
    Process management related errors.
    
    Additional Attributes:
        process_id (int): Process ID
        command (str): Command that was executed
        exit_code (Optional[int]): Process exit code
        runtime_stats (Dict[str, Any]): Process runtime statistics
    """
    def __init__(
        self,
        message: str,
        process_id: int,
        command: str,
        exit_code: Optional[int] = None,
        runtime_stats: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
        retryable: bool = True,
        log_error: bool = True
    ) -> None:
        details = {
            "process_id": process_id,
            "command": command,
            "exit_code": exit_code,
            "runtime_stats": runtime_stats or {},
            "error_type": type(original_error).__name__ if original_error else None,
            "error_message": str(original_error) if original_error else None
        }
        
        super().__init__(
            message,
            code="PROCESS_ERROR",
            details=details,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.SYSTEM,
            source="ProcessManager",
            retryable=retryable,
            log_error=log_error,
            tags=["process", f"pid_{process_id}"]
        )

def error_handler(error_category: str = ErrorCategory.SYSTEM, retryable: bool = True):
    """
    Decorator for handling errors in functions.
    
    Args:
        error_category (str): Category of errors to handle
        retryable (bool): Whether errors should be retryable
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except BaseError as e:
                e.error_detail.category = error_category
                e.retryable = retryable
                raise
            except Exception as e:
                error = BaseError(
                    message=str(e),
                    category=error_category,
                    retryable=retryable,
                    details={"args": args, "kwargs": kwargs}
                )
                raise error from e
        return wrapper
    return decorator

# Error tracking singleton
class ErrorTracker:
    """
    Singleton class for tracking and analyzing errors.
    """
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.errors: Dict[str, ErrorDetail] = {}
                cls._instance.error_counts: Dict[str, int] = {}
                cls._instance.error_rates: Dict[str, float] = {}
        return cls._instance
    
    def track_error(self, error: Union[BaseError, ErrorDetail]) -> None:
        """Track an error occurrence."""
        with self._lock:
            if isinstance(error, BaseError):
                error_detail = error.error_detail
            else:
                error_detail = error
                
            self.errors[error_detail.correlation_id] = error_detail
            self.error_counts[error_detail.code] = self.error_counts.get(error_detail.code, 0) + 1
            
            # Update error rates
            now = datetime.utcnow()
            if error_detail.metrics.first_occurrence:
                time_diff = (now - error_detail.metrics.first_occurrence).total_seconds()
                self.error_rates[error_detail.code] = self.error_counts[error_detail.code] / time_diff
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self._lock:
            return {
                "total_errors": len(self.errors),
                "error_counts": self.error_counts.copy(),
                "error_rates": self.error_rates.copy(),
                "errors_by_severity": self._group_errors_by("severity"),
                "errors_by_category": self._group_errors_by("category"),
            }
    
    def _group_errors_by(self, attribute: str) -> Dict[str, int]:
        """Group errors by a specific attribute."""
        groups: Dict[str, int] = {}
        for error in self.errors.values():
            key = getattr(error, attribute)
            groups[key] = groups.get(key, 0) + 1
        return groups

# Initialize error tracker
error_tracker = ErrorTracker()

# Export all exception classes and utilities
__all__ = [
    'ErrorSeverity',
    'ErrorCategory',
    'ErrorContext',
    'ErrorMetrics',
    'ErrorDetail',
    'BaseError',
    'WindowsError',
    'ResourceError',
    'CacheError',
    'FileSystemError',
    'NetworkError',
    'SecurityError',
    'ProcessError',
    'error_handler',
    'ErrorTracker',
    'error_tracker',
]

# Last modified: 2025-02-03 17:42:17 UTC
# Modified by: drphon
# End of exceptions.py