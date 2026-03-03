"""PageIndex: Vectorless, reasoning-based RAG using hierarchical document indexing."""

from pageindex.batch import DoclingServeClient, process_folder, process_folder_sync
from pageindex.config import PageIndexConfig
from pageindex.markdown.processor import md_to_tree
from pageindex.pdf.processor import PageIndexProcessor, page_index
from pageindex.repo import index_repository, index_repository_sync

__version__ = "0.1.0"

__all__ = [
    "DoclingServeClient",
    "PageIndexConfig",
    "PageIndexProcessor",
    "__version__",
    "index_repository",
    "index_repository_sync",
    "md_to_tree",
    "page_index",
    "process_folder",
    "process_folder_sync",
]
