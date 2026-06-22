import urllib


def is_url(location: str) -> bool:
    """Return True if `location` is a url. False otherwise."""
    return urllib.parse.urlparse(location).scheme in ["http", "https"]
