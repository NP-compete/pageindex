"""Tests for utility functions."""

from pageindex.utils import (
    convert_page_to_int,
    convert_physical_index_to_int,
    extract_json,
    get_json_content,
    sanitize_filename,
)


class TestExtractJson:
    """Tests for extract_json function."""

    def test_plain_json(self):
        """Test extracting plain JSON."""
        content = '{"key": "value"}'
        result = extract_json(content)
        assert result == {"key": "value"}

    def test_json_in_code_block(self):
        """Test extracting JSON from code block."""
        content = '```json\n{"key": "value"}\n```'
        result = extract_json(content)
        assert result == {"key": "value"}

    def test_json_with_none(self):
        """Test extracting JSON with Python None."""
        content = '{"key": None}'
        result = extract_json(content)
        assert result == {"key": None}

    def test_invalid_json(self):
        """Test handling invalid JSON."""
        content = "not json at all"
        result = extract_json(content)
        assert result == {}


class TestGetJsonContent:
    """Tests for get_json_content function."""

    def test_with_code_block(self):
        """Test extracting content from code block."""
        response = '```json\n{"test": true}\n```'
        result = get_json_content(response)
        assert result == '{"test": true}'

    def test_without_code_block(self):
        """Test with plain content."""
        response = '{"test": true}'
        result = get_json_content(response)
        assert result == '{"test": true}'


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_clean_filename(self):
        """Test with clean filename."""
        result = sanitize_filename("document.pdf")
        assert result == "document.pdf"

    def test_filename_with_slash(self):
        """Test filename with forward slash."""
        result = sanitize_filename("path/to/file.pdf")
        assert result == "path-to-file.pdf"

    def test_filename_with_backslash(self):
        """Test filename with backslash."""
        result = sanitize_filename("path\\to\\file.pdf")
        assert result == "path-to-file.pdf"

    def test_custom_replacement(self):
        """Test with custom replacement character."""
        result = sanitize_filename("path/file.pdf", replacement="_")
        assert result == "path_file.pdf"


class TestConvertPhysicalIndexToInt:
    """Tests for convert_physical_index_to_int function."""

    def test_list_with_physical_index(self):
        """Test converting list items."""
        data = [{"physical_index": "<physical_index_5>"}]
        result = convert_physical_index_to_int(data)
        assert result[0]["physical_index"] == 5

    def test_string_physical_index(self):
        """Test converting string directly."""
        data = "<physical_index_10>"
        result = convert_physical_index_to_int(data)
        assert result == 10

    def test_physical_index_underscore_format(self):
        """Test alternative format."""
        data = [{"physical_index": "physical_index_15"}]
        result = convert_physical_index_to_int(data)
        assert result[0]["physical_index"] == 15


class TestConvertPageToInt:
    """Tests for convert_page_to_int function."""

    def test_string_page(self):
        """Test converting string page to int."""
        data = [{"page": "5"}]
        result = convert_page_to_int(data)
        assert result[0]["page"] == 5

    def test_int_page(self):
        """Test with already int page."""
        data = [{"page": 10}]
        result = convert_page_to_int(data)
        assert result[0]["page"] == 10

    def test_invalid_page(self):
        """Test with invalid page string."""
        data = [{"page": "not a number"}]
        result = convert_page_to_int(data)
        assert result[0]["page"] == "not a number"
