#!/usr/bin/env python
"""
Download RFC documents from rfc-editor.org.

This module downloads actual RFC .txt files which contain the
authoritative specifications for Internet protocols.
"""

import os
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Set
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)


class RFCDownloader:
    """Download RFC documents from rfc-editor.org."""
    
    # Base URLs for RFC documents
    RFC_BASE_URL = "https://www.rfc-editor.org/rfc"
    RFC_TXT_URL = "https://www.rfc-editor.org/rfc/rfc{num}.txt"
    
    # Extended RFC list - includes major protocol RFCs from 1-9500
    # This covers most important Internet standards
    KEY_RFCS = [
        # Core Internet protocols (700s-800s)
        *range(760, 800),   # Early core protocols
        791, 792, 793,      # IP, ICMP, TCP
        
        # 800s-900s - More fundamentals
        *range(820, 880),
        *range(950, 1000),
        
        # 1000s - DNS, SMTP evolution
        *range(1000, 1100),
        
        # 1100s-1200s - Requirements docs
        *range(1100, 1200),
        
        # 1300s-1500s - Various protocols
        *range(1300, 1500),
        
        # 1700s-1900s - More protocols
        *range(1700, 1950),
        
        # 2000s - MIME, HTTP, etc.
        *range(2000, 2200),
        *range(2300, 2500),
        2616,  # HTTP/1.1
        *range(2700, 2900),
        
        # 3000s - More modern protocols
        *range(3000, 3100),
        *range(3200, 3400),
        *range(3500, 3700),
        3986,  # URI
        
        # 4000s - TLS, SSH, BGP updates
        *range(4000, 4300),
        *range(4500, 4700),
        
        # 5000s - Major updates
        *range(5000, 5400),
        *range(5700, 6000),
        
        # 6000s - Modern standards
        *range(6000, 6200),
        *range(6400, 6800),
        
        # 7000s - HTTP/2, JSON, etc.
        *range(7000, 7600),
        
        # 8000s - TLS 1.3, modern updates
        *range(8000, 8500),
        
        # 9000s - HTTP/3, latest standards
        *range(9000, 9200),
    ]
    
    def __init__(self, output_dir: str, delay: float = 0.5):
        """
        Initialize the RFC downloader.
        
        Args:
            output_dir: Directory to store downloaded RFCs
            delay: Delay between downloads in seconds (be nice to server)
        """
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.downloaded: Set[int] = set()
        
    def download_rfc(self, rfc_num: int) -> Tuple[Optional[Path], str]:
        """
        Download a single RFC.
        
        Returns:
            Tuple of (path, status_message)
            path is None if download failed
        """
        url = self.RFC_TXT_URL.format(num=rfc_num)
        filename = f"rfc{rfc_num}.txt"
        output_path = self.output_dir / filename
        
        # Check if already downloaded
        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path, "exists"
        
        try:
            # Make request with proper headers
            request = Request(
                url,
                headers={
                    'User-Agent': 'MyPT-CorpusBuilder/1.0 (Educational/Research)',
                    'Accept': 'text/plain',
                }
            )
            
            with urlopen(request, timeout=30) as response:
                content = response.read()
            
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Write to file
            with open(output_path, 'wb') as f:
                f.write(content)
            
            self.downloaded.add(rfc_num)
            return output_path, "downloaded"
            
        except HTTPError as e:
            if e.code == 404:
                return None, f"not_found (RFC {rfc_num} doesn't exist)"
            return None, f"http_error_{e.code}"
        except URLError as e:
            return None, f"url_error: {e.reason}"
        except Exception as e:
            return None, f"error: {e}"
    
    def download_key_rfcs(self, 
                          progress_callback=None,
                          max_rfcs: Optional[int] = None) -> List[Tuple[int, Path, str]]:
        """
        Download all key RFCs.
        
        Args:
            progress_callback: Optional callback(current, total, rfc_num, status)
            max_rfcs: Optional limit on number of RFCs to download
            
        Returns:
            List of (rfc_num, path, status) tuples
        """
        results = []
        rfcs_to_download = self.KEY_RFCS[:max_rfcs] if max_rfcs else self.KEY_RFCS
        total = len(rfcs_to_download)
        
        for i, rfc_num in enumerate(rfcs_to_download):
            path, status = self.download_rfc(rfc_num)
            results.append((rfc_num, path, status))
            
            if progress_callback:
                progress_callback(i + 1, total, rfc_num, status)
            
            # Rate limiting (only if we actually downloaded)
            if status == "downloaded" and i < total - 1:
                time.sleep(self.delay)
        
        return results
    
    def download_range(self, 
                       start: int, 
                       end: int,
                       progress_callback=None) -> List[Tuple[int, Path, str]]:
        """
        Download RFCs in a numeric range.
        
        Args:
            start: Starting RFC number
            end: Ending RFC number (inclusive)
            progress_callback: Optional callback(current, total, rfc_num, status)
            
        Returns:
            List of (rfc_num, path, status) tuples
        """
        results = []
        total = end - start + 1
        
        for i, rfc_num in enumerate(range(start, end + 1)):
            path, status = self.download_rfc(rfc_num)
            results.append((rfc_num, path, status))
            
            if progress_callback:
                progress_callback(i + 1, total, rfc_num, status)
            
            # Rate limiting (only if we actually downloaded)
            if status == "downloaded" and rfc_num < end:
                time.sleep(self.delay)
        
        return results
    
    def get_stats(self) -> dict:
        """Get download statistics."""
        existing = list(self.output_dir.glob("rfc*.txt")) if self.output_dir.exists() else []
        total_size = sum(f.stat().st_size for f in existing)
        
        return {
            "downloaded_this_session": len(self.downloaded),
            "total_files": len(existing),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
        }


def download_rfcs(output_dir: str, 
                  max_rfcs: Optional[int] = None,
                  verbose: bool = True) -> dict:
    """
    Convenience function to download key RFCs.
    
    Args:
        output_dir: Directory to store downloaded RFCs
        max_rfcs: Optional limit on number of RFCs
        verbose: Print progress
        
    Returns:
        Statistics dictionary
    """
    downloader = RFCDownloader(output_dir)
    
    def progress(current, total, rfc_num, status):
        if verbose:
            print(f"  [{current}/{total}] RFC {rfc_num}: {status}")
    
    if verbose:
        print(f"Downloading key RFCs to {output_dir}")
    
    results = downloader.download_key_rfcs(
        progress_callback=progress if verbose else None,
        max_rfcs=max_rfcs
    )
    
    successful = sum(1 for _, path, _ in results if path is not None)
    
    stats = downloader.get_stats()
    stats["requested"] = len(results)
    stats["successful"] = successful
    
    if verbose:
        print(f"\nDownloaded {successful}/{len(results)} RFCs")
        print(f"Total: {stats['total_files']} files, {stats['total_size_mb']:.1f} MB")
    
    return stats


if __name__ == '__main__':
    import sys
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./rfc_downloads"
    max_rfcs = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    download_rfcs(output_dir, max_rfcs=max_rfcs)

