"""Tests for tree operations."""

from pageindex.tree import (
    add_preface_if_needed,
    format_structure,
    get_leaf_nodes,
    get_nodes,
    is_leaf_node,
    list_to_tree,
    remove_page_number,
    structure_to_list,
    write_node_id,
)


class TestWriteNodeId:
    """Tests for write_node_id function."""

    def test_single_node(self):
        """Test assigning ID to a single node."""
        data = {"title": "Test"}
        write_node_id(data)
        assert data["node_id"] == "0000"

    def test_nested_nodes(self):
        """Test assigning IDs to nested nodes."""
        data = {
            "title": "Root",
            "nodes": [
                {"title": "Child 1", "nodes": []},
                {"title": "Child 2", "nodes": []},
            ],
        }
        write_node_id(data)
        assert data["node_id"] == "0000"
        assert data["nodes"][0]["node_id"] == "0001"
        assert data["nodes"][1]["node_id"] == "0002"

    def test_list_of_nodes(self):
        """Test assigning IDs to a list of nodes."""
        data = [
            {"title": "Node 1"},
            {"title": "Node 2"},
        ]
        write_node_id(data)
        assert data[0]["node_id"] == "0000"
        assert data[1]["node_id"] == "0001"


class TestGetNodes:
    """Tests for get_nodes function."""

    def test_flat_structure(self):
        """Test getting nodes from flat structure."""
        structure = {"title": "Root", "nodes": []}
        nodes = get_nodes(structure)
        assert len(nodes) == 1
        assert nodes[0]["title"] == "Root"

    def test_nested_structure(self):
        """Test getting nodes from nested structure."""
        structure = {
            "title": "Root",
            "nodes": [
                {"title": "Child 1", "nodes": []},
                {"title": "Child 2", "nodes": []},
            ],
        }
        nodes = get_nodes(structure)
        assert len(nodes) == 3
        titles = [n["title"] for n in nodes]
        assert "Root" in titles
        assert "Child 1" in titles
        assert "Child 2" in titles


class TestStructureToList:
    """Tests for structure_to_list function."""

    def test_single_node(self):
        """Test converting single node to list."""
        structure = {"title": "Root"}
        result = structure_to_list(structure)
        assert len(result) == 1

    def test_nested_structure(self):
        """Test converting nested structure to list."""
        structure = {
            "title": "Root",
            "nodes": [{"title": "Child"}],
        }
        result = structure_to_list(structure)
        assert len(result) == 2


class TestGetLeafNodes:
    """Tests for get_leaf_nodes function."""

    def test_single_leaf(self):
        """Test getting leaf from single node."""
        structure = {"title": "Leaf", "nodes": []}
        leaves = get_leaf_nodes(structure)
        assert len(leaves) == 1
        assert leaves[0]["title"] == "Leaf"

    def test_nested_leaves(self):
        """Test getting leaves from nested structure."""
        structure = {
            "title": "Root",
            "nodes": [
                {"title": "Leaf 1", "nodes": []},
                {"title": "Leaf 2", "nodes": []},
            ],
        }
        leaves = get_leaf_nodes(structure)
        assert len(leaves) == 2
        titles = [leaf["title"] for leaf in leaves]
        assert "Leaf 1" in titles
        assert "Leaf 2" in titles


class TestIsLeafNode:
    """Tests for is_leaf_node function."""

    def test_leaf_node(self):
        """Test identifying a leaf node."""
        data = {"node_id": "0001", "title": "Leaf", "nodes": []}
        assert is_leaf_node(data, "0001") is True

    def test_non_leaf_node(self):
        """Test identifying a non-leaf node."""
        data = {
            "node_id": "0001",
            "title": "Parent",
            "nodes": [{"node_id": "0002", "title": "Child", "nodes": []}],
        }
        assert is_leaf_node(data, "0001") is False
        assert is_leaf_node(data, "0002") is True


class TestListToTree:
    """Tests for list_to_tree function."""

    def test_flat_list(self):
        """Test converting flat list to tree."""
        data = [
            {"structure": "1", "title": "Section 1", "start_index": 1, "end_index": 5},
            {"structure": "2", "title": "Section 2", "start_index": 6, "end_index": 10},
        ]
        tree = list_to_tree(data)
        assert len(tree) == 2
        assert tree[0]["title"] == "Section 1"
        assert tree[1]["title"] == "Section 2"

    def test_nested_list(self):
        """Test converting nested list to tree."""
        data = [
            {"structure": "1", "title": "Section 1", "start_index": 1, "end_index": 10},
            {"structure": "1.1", "title": "Subsection 1.1", "start_index": 2, "end_index": 5},
            {"structure": "1.2", "title": "Subsection 1.2", "start_index": 6, "end_index": 10},
        ]
        tree = list_to_tree(data)
        assert len(tree) == 1
        assert tree[0]["title"] == "Section 1"
        assert len(tree[0]["nodes"]) == 2


class TestAddPrefaceIfNeeded:
    """Tests for add_preface_if_needed function."""

    def test_no_preface_needed(self):
        """Test when no preface is needed."""
        data = [{"physical_index": 1, "title": "Chapter 1"}]
        result = add_preface_if_needed(data)
        assert len(result) == 1

    def test_preface_needed(self):
        """Test when preface is needed."""
        data = [{"physical_index": 5, "title": "Chapter 1"}]
        result = add_preface_if_needed(data)
        assert len(result) == 2
        assert result[0]["title"] == "Preface"
        assert result[0]["physical_index"] == 1

    def test_empty_list(self):
        """Test with empty list."""
        data = []
        result = add_preface_if_needed(data)
        assert result == []


class TestRemovePageNumber:
    """Tests for remove_page_number function."""

    def test_remove_page_number(self):
        """Test removing page_number field."""
        data = {"title": "Test", "page_number": 5}
        result = remove_page_number(data)
        assert "page_number" not in result
        assert result["title"] == "Test"

    def test_nested_remove(self):
        """Test removing page_number from nested structure."""
        data = {
            "title": "Root",
            "page_number": 1,
            "nodes": [{"title": "Child", "page_number": 2}],
        }
        result = remove_page_number(data)
        assert "page_number" not in result
        assert "page_number" not in result["nodes"][0]


class TestFormatStructure:
    """Tests for format_structure function."""

    def test_reorder_keys(self):
        """Test reordering dictionary keys."""
        structure = {"b": 2, "a": 1, "c": 3}
        result = format_structure(structure, order=["a", "b", "c"])
        keys = list(result.keys())
        assert keys == ["a", "b", "c"]

    def test_no_order(self):
        """Test with no order specified."""
        structure = {"b": 2, "a": 1}
        result = format_structure(structure, order=None)
        assert result == structure
