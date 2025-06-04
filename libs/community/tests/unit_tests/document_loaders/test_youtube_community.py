import pytest
from langchain_core.documents import Document

from langchain_community.document_loaders import YoutubeCommunityLoader
from langchain_community.document_loaders.youtube_community import (
    _parse_community_post_url,
)


@pytest.mark.parametrize(
    "youtube_community_url, expected_post_id",
    [
        (
            "https://www.youtube.com/post/UgkxCWeKPiIOLsnh_5a0mpHWywRz",
            "UgkxCWeKPiIOLsnh_5a0mpHWywRz",
        ),
        (
            "https://youtube.com/post/UgkxCWeKPiIOLsnh_5a0mpHWywRz",
            "UgkxCWeKPiIOLsnh_5a0mpHWywRz",
        ),
        (
            "https://m.youtube.com/post/UgkxCWeKPiIOLsnh_5a0mpHWywRz",
            "UgkxCWeKPiIOLsnh_5a0mpHWywRz",
        ),
    ],
)
def test_community_post_id_extraction_valid(
    youtube_community_url: str, expected_post_id: str
) -> None:
    """Test that community post IDs are correctly extracted from valid URLs."""
    assert (
        YoutubeCommunityLoader.extract_post_id(youtube_community_url)
        == expected_post_id
    )


@pytest.mark.parametrize(
    "youtube_community_url",
    [
        "https://www.youtube.com/watch?v=abc123",  # Not a post URL
        "https://www.youtube.com/channel/UC_x5XG1OV2P6uZZ5FSM9Ttw",  # Channel URL
        "https://example.com/post/UgkxCWeKPiIOLsnh_5a0mpHWywRz",  # Wrong domain
    ],
)
def test_community_post_id_extraction_invalid(youtube_community_url: str) -> None:
    """Test that invalid URLs raise ValueError."""
    with pytest.raises(ValueError):
        YoutubeCommunityLoader.extract_post_id(youtube_community_url)


def test_parse_community_post_url_with_valid_urls() -> None:
    """Test parsing valid community post URLs."""
    valid_urls = [
        "https://www.youtube.com/post/PostID123",
        "https://youtube.com/post/PostID456",
        "https://m.youtube.com/post/PostID789",
    ]

    expected_ids = ["PostID123", "PostID456", "PostID789"]

    for url, expected_id in zip(valid_urls, expected_ids):
        assert _parse_community_post_url(url) == expected_id


def test_parse_community_post_url_with_invalid_urls() -> None:
    """Test parsing invalid URLs raises ValueError."""
    invalid_urls = [
        "https://www.youtube.com/watch?v=abc123",  # Not a post URL
        "https://example.com/post/PostID",  # Wrong domain
        "ftp://youtube.com/post/PostID",  # Wrong scheme
        "https://youtube.com/channel/UC123/videos",  # Wrong path
        "https://youtube.com/channel/UC123/community",  # Old format
    ]

    for url in invalid_urls:
        with pytest.raises(ValueError):
            _parse_community_post_url(url)


def test_youtube_community_loader_initialization() -> None:
    """Test various ways to initialize YoutubeCommunityLoader."""
    # Test with post ID
    loader1 = YoutubeCommunityLoader("UgkxCWeKPiIOLsnh_5a0mpHWywRz")
    assert loader1.post_id == "UgkxCWeKPiIOLsnh_5a0mpHWywRz"
    assert loader1.continue_on_failure is False

    # Test with continue_on_failure
    loader2 = YoutubeCommunityLoader("PostID123", continue_on_failure=True)
    assert loader2.post_id == "PostID123"
    assert loader2.continue_on_failure is True


def test_from_youtube_community_url_class_method() -> None:
    """Test the from_youtube_community_url class method."""
    url = "https://www.youtube.com/post/UgkxCWeKPiIOLsnh_5a0mpHWywRz"

    loader = YoutubeCommunityLoader.from_youtube_community_url(url)
    assert loader.post_id == "UgkxCWeKPiIOLsnh_5a0mpHWywRz"

    # Test with additional parameters
    loader2 = YoutubeCommunityLoader.from_youtube_community_url(
        url, continue_on_failure=True
    )
    assert loader2.continue_on_failure is True


def test_from_youtube_community_url_invalid_url() -> None:
    """Test that invalid URLs raise ValueError."""
    with pytest.raises(ValueError, match="Invalid URL netloc"):
        YoutubeCommunityLoader.from_youtube_community_url("https://example.com/invalid")

    with pytest.raises(ValueError, match="Invalid URL path"):
        YoutubeCommunityLoader.from_youtube_community_url("https://youtube.com/watch?v=abc123")


def test_create_document() -> None:
    """Test the _create_document method."""
    loader = YoutubeCommunityLoader("TestPost123")

    # Mock post data
    post_data = {
        "post_link": "https://www.youtube.com/post/TestPost123",
        "post_id": "TestPost123",
        "text_content": "This is a test community post",
        "time_since": "2 hours ago",
        "time_of_download": "2023-01-01 12:00:00 UTC",
        "video_link": "https://www.youtube.com/watch?v=abc123",
        "image_links": ["https://i.ytimg.com/image1.jpg", "https://i.ytimg.com/image2.jpg"],
        "poll_content": "Poll: Which option do you prefer? A: 45%, B: 55%",
    }

    doc = loader._create_document(post_data)

    assert isinstance(doc, Document)
    assert doc.page_content == "This is a test community post"
    assert doc.metadata["source"] == "https://www.youtube.com/post/TestPost123"
    assert doc.metadata["post_id"] == "TestPost123"
    assert doc.metadata["time_since"] == "2 hours ago"
    assert doc.metadata["time_of_download"] == "2023-01-01 12:00:00 UTC"
    assert doc.metadata["video_link"] == "https://www.youtube.com/watch?v=abc123"
    assert doc.metadata["image_links"] == ["https://i.ytimg.com/image1.jpg", "https://i.ytimg.com/image2.jpg"]
    assert doc.metadata["poll_content"] == "Poll: Which option do you prefer? A: 45%, B: 55%"


def test_create_document_minimal() -> None:
    """Test _create_document with minimal data."""
    loader = YoutubeCommunityLoader("TestPost123")

    # Minimal post data
    post_data = {
        "post_link": "https://www.youtube.com/post/TestPost123",
        "post_id": "TestPost123",
        "text_content": "Minimal post",
        "time_since": "Unknown",
        "time_of_download": "2023-01-01 12:00:00 UTC",
    }

    doc = loader._create_document(post_data)

    assert doc.page_content == "Minimal post"
    assert doc.metadata["source"] == "https://www.youtube.com/post/TestPost123"
    assert doc.metadata["post_id"] == "TestPost123"
    assert doc.metadata["time_since"] == "Unknown"
    assert doc.metadata["time_of_download"] == "2023-01-01 12:00:00 UTC"
    # Optional fields should not be present if empty
    assert "video_link" not in doc.metadata
    assert "image_links" not in doc.metadata
    assert "poll_content" not in doc.metadata


@pytest.mark.skip(reason="Requires external network access")
def test_load_integration() -> None:
    """Test loading with actual network requests (integration test)."""
    loader = YoutubeCommunityLoader("UgkxCWeKPiIOLsnh_5a0mpHWywRz", continue_on_failure=True)

    # This would normally require actual network requests
    # Skip for unit tests, but structure shows how it would work
    documents = loader.load()

    assert isinstance(documents, list)
    if documents:  # If any documents returned
        assert isinstance(documents[0], Document)
        assert documents[0].metadata["post_id"] == "UgkxCWeKPiIOLsnh_5a0mpHWywRz"


def test_loader_continue_on_failure() -> None:
    """Test that continue_on_failure parameter works correctly."""
    loader = YoutubeCommunityLoader("TestPost123", continue_on_failure=True)
    assert loader.continue_on_failure is True

    loader_strict = YoutubeCommunityLoader("TestPost123", continue_on_failure=False)
    assert loader_strict.continue_on_failure is False
