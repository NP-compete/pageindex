"""Tests for configuration."""

from pageindex.config import ConfigLoader, PageIndexConfig


class TestPageIndexConfig:
    """Tests for PageIndexConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PageIndexConfig()
        assert config.location == "us-central1"
        assert config.model == "gemini-1.5-flash"
        assert config.toc_check_page_num == 20
        assert config.max_page_num_each_node == 10
        assert config.max_token_num_each_node == 20000
        assert config.if_add_node_id == "yes"
        assert config.if_add_node_summary == "yes"
        assert config.if_add_doc_description == "no"
        assert config.if_add_node_text == "no"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PageIndexConfig(
            project_id="test-project",
            location="europe-west1",
            model="gemini-1.5-pro",
            toc_check_page_num=30,
        )
        assert config.project_id == "test-project"
        assert config.location == "europe-west1"
        assert config.model == "gemini-1.5-pro"
        assert config.toc_check_page_num == 30

    def test_yes_no_validation(self):
        """Test yes/no field validation."""
        config = PageIndexConfig(if_add_node_id="no")
        assert config.if_add_node_id == "no"


class TestConfigLoader:
    """Tests for ConfigLoader."""

    def test_load_with_none(self):
        """Test loading with no user options."""
        loader = ConfigLoader()
        config = loader.load(None)
        assert isinstance(config, PageIndexConfig)

    def test_load_with_dict(self):
        """Test loading with dictionary options."""
        loader = ConfigLoader()
        config = loader.load({"project_id": "my-project", "model": "gemini-1.5-pro"})
        assert config.project_id == "my-project"
        assert config.model == "gemini-1.5-pro"

    def test_load_with_config(self):
        """Test loading with existing config."""
        loader = ConfigLoader()
        existing = PageIndexConfig(project_id="existing-project")
        config = loader.load(existing)
        assert config.project_id == "existing-project"
