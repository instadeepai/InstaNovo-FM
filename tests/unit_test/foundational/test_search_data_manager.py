"""Unit tests for SearchDataManager.

Tests the functionality of loading and querying metadata from search_data.xlsx.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from instanovo_fm.data.search_data_manager import (
    SearchDataManager,
    _extract_lookup_key,
    _normalize_path,
    create_search_data_manager,
)


class TestPathNormalization:
    """Test path normalization utilities."""
    
    def test_normalize_path_forward_slashes(self):
        """Test that backslashes are converted to forward slashes."""
        path = r"C:\Users\data\file.mzML"
        expected = "C:/Users/data/file.mzML"
        assert _normalize_path(path) == expected
    
    def test_normalize_path_mixed_slashes(self):
        """Test mixed slash normalization."""
        path = r"C:\Users/data\file.mzML"
        expected = "C:/Users/data/file.mzML"
        assert _normalize_path(path) == expected
    
    def test_normalize_path_already_normalized(self):
        """Test that already normalized paths are unchanged."""
        path = "C:/Users/data/file.mzML"
        expected = "C:/Users/data/file.mzML"
        assert _normalize_path(path) == expected


class TestLookupKeyExtraction:
    """Test lookup key extraction from file paths."""
    
    def test_extract_basic_path(self):
        """Test extraction from basic path."""
        path = "/data/ProjectABC/sample01.mzML"
        parent, filename = _extract_lookup_key(path)
        assert parent == "ProjectABC"
        assert filename == "sample01"
    
    def test_extract_with_double_extension(self):
        """Test extraction with .mzML.gz extension."""
        path = "/data/ProjectABC/sample01.mzML.gz"
        parent, filename = _extract_lookup_key(path)
        assert parent == "ProjectABC"
        assert filename == "sample01"
    
    def test_extract_with_shard_suffix(self):
        """Test extraction with shard suffix."""
        path = "/data/ProjectABC/sample01_0001-0002.mzML.gz"
        parent, filename = _extract_lookup_key(path)
        assert parent == "ProjectABC"
        assert filename == "sample01"
    
    def test_extract_windows_path(self):
        """Test extraction from Windows path."""
        path = r"\\server\share\ProjectXYZ\sample02_0003-0004.mzML.gz"
        parent, filename = _extract_lookup_key(path)
        assert parent == "ProjectXYZ"
        assert filename == "sample02"
    
    def test_extract_empty_path(self):
        """Test extraction from empty path."""
        path = ""
        parent, filename = _extract_lookup_key(path)
        assert parent == "unknown"
        assert filename == "unknown"
    
    def test_extract_different_extensions(self):
        """Test extraction with various MS file extensions."""
        test_cases = [
            ("/data/proj/file.raw", "proj", "file"),
            ("/data/proj/file.wiff", "proj", "file"),
            ("/data/proj/file.d", "proj", "file"),
            ("/data/proj/file.mgf", "proj", "file"),
            ("/data/proj/file.mzXML", "proj", "file"),
        ]
        for path, expected_parent, expected_filename in test_cases:
            parent, filename = _extract_lookup_key(path)
            assert parent == expected_parent
            assert filename == expected_filename


class TestSearchDataManager:
    """Test SearchDataManager class."""
    
    @pytest.fixture
    def sample_excel_file(self):
        """Create a temporary Excel file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
            excel_path = f.name
        
        # Create sample data
        data = {
            'file path': [
                '/data/ProjectA/sample01.mzML.gz',
                '/data/ProjectA/sample02_0001-0002.mzML.gz',
                '/data/ProjectB/sample03.raw',
                '',  # Empty path
            ],
            'experiment_name': ['Exp1', 'Exp2', 'Exp3', 'Exp4'],
            'collision_energy': [25.0, 30.0, 28.5, 35.0],
            'instrument': ['QE', 'QE', 'Orbitrap', 'Orbitrap'],
        }
        df = pd.DataFrame(data)
        df.to_excel(excel_path, index=False)
        
        yield excel_path
        
        # Cleanup
        os.unlink(excel_path)
    
    def test_manager_initialization(self, sample_excel_file):
        """Test manager initialization."""
        manager = SearchDataManager(sample_excel_file)
        assert manager.file_path == sample_excel_file
        assert not manager.is_loaded
    
    def test_manager_load(self, sample_excel_file):
        """Test loading Excel file."""
        manager = SearchDataManager(sample_excel_file)
        manager.load()
        
        assert manager.is_loaded
        assert len(manager._lookup) == 3  # 4 rows - 1 empty = 3 indexed
        assert manager._unindexed_rows == 1
    
    def test_manager_load_missing_file(self):
        """Test loading non-existent file."""
        manager = SearchDataManager("nonexistent.xlsx")
        with pytest.raises(FileNotFoundError):
            manager.load()
    
    def test_manager_load_missing_column(self):
        """Test loading file without required column."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
            excel_path = f.name
        
        try:
            # Create file without "file path" column
            df = pd.DataFrame({'wrong_column': [1, 2, 3]})
            df.to_excel(excel_path, index=False)
            
            manager = SearchDataManager(excel_path)
            with pytest.raises(ValueError, match="Required column"):
                manager.load()
        finally:
            os.unlink(excel_path)
    
    def test_get_metadata_exact_match(self, sample_excel_file):
        """Test getting metadata with exact path match."""
        manager = SearchDataManager(sample_excel_file)
        manager.load()
        
        # Test exact match
        metadata = manager.get_metadata('/data/ProjectA/sample01.mzML.gz')
        assert metadata['experiment_name'] == 'Exp1'
        assert metadata['collision_energy'] == 25.0
        assert metadata['instrument'] == 'QE'
    
    def test_get_metadata_with_shard(self, sample_excel_file):
        """Test getting metadata with shard suffix."""
        manager = SearchDataManager(sample_excel_file)
        manager.load()
        
        # Test match with shard
        metadata = manager.get_metadata('/data/ProjectA/sample02_0001-0002.mzML.gz')
        assert metadata['experiment_name'] == 'Exp2'
        assert metadata['collision_energy'] == 30.0
    
    def test_get_metadata_extension_variation(self, sample_excel_file):
        """Test getting metadata with different extension."""
        manager = SearchDataManager(sample_excel_file)
        manager.load()
        
        # Should match based on parent/filename even with different extension
        # sample01.mzML.gz in Excel, querying with sample01.mzML
        metadata = manager.get_metadata('/data/ProjectA/sample01.mzML')
        assert metadata['experiment_name'] == 'Exp1'
    
    def test_get_metadata_not_found(self, sample_excel_file):
        """Test getting metadata for non-existent path."""
        manager = SearchDataManager(sample_excel_file)
        manager.load()
        
        metadata = manager.get_metadata('/data/ProjectZ/unknown.mzML')
        assert metadata == {}
    
    def test_get_metadata_not_loaded(self, sample_excel_file):
        """Test getting metadata before loading."""
        manager = SearchDataManager(sample_excel_file)
        metadata = manager.get_metadata('/data/ProjectA/sample01.mzML.gz')
        assert metadata == {}
    
    def test_get_metadata_batch(self, sample_excel_file):
        """Test batch metadata retrieval."""
        manager = SearchDataManager(sample_excel_file)
        manager.load()
        
        paths = [
            '/data/ProjectA/sample01.mzML.gz',
            '/data/ProjectB/sample03.raw',
            '/data/ProjectZ/unknown.mzML',
        ]
        
        results = manager.get_metadata_batch(paths)
        assert len(results) == 3
        assert results[0]['experiment_name'] == 'Exp1'
        assert results[1]['experiment_name'] == 'Exp3'
        assert results[2] == {}
    
    def test_summary_not_loaded(self, sample_excel_file):
        """Test summary before loading."""
        manager = SearchDataManager(sample_excel_file)
        summary = manager.summary()
        
        assert summary['loaded'] is False
        assert summary['file_path'] == sample_excel_file
    
    def test_summary_loaded(self, sample_excel_file):
        """Test summary after loading."""
        manager = SearchDataManager(sample_excel_file)
        manager.load()
        summary = manager.summary()
        
        assert summary['loaded'] is True
        assert summary['total_rows'] == 4
        assert summary['indexed_rows'] == 3
        assert summary['unindexed_rows'] == 1
        assert 'file path' in summary['columns']
    
    def test_get_available_columns(self, sample_excel_file):
        """Test getting available columns."""
        manager = SearchDataManager(sample_excel_file)
        
        # Before loading
        assert manager.get_available_columns() == []
        
        # After loading
        manager.load()
        columns = manager.get_available_columns()
        assert 'file path' in columns
        assert 'experiment_name' in columns
        assert 'collision_energy' in columns
        assert 'instrument' in columns


class TestFactoryFunction:
    """Test create_search_data_manager factory function."""
    
    def test_create_disabled(self):
        """Test creating manager with disabled flag."""
        config = {'use_search_data': False}
        manager = create_search_data_manager(config)
        assert manager is None
    
    def test_create_no_path(self):
        """Test creating manager without path."""
        config = {'use_search_data': True}
        manager = create_search_data_manager(config)
        assert manager is None
    
    def test_create_with_invalid_path(self):
        """Test creating manager with invalid path."""
        config = {
            'use_search_data': True,
            'search_data_path': 'nonexistent.xlsx',
        }
        manager = create_search_data_manager(config)
        assert manager is None  # Should handle error gracefully
    
    def test_create_with_valid_path(self, sample_excel_file=None):
        """Test creating manager with valid configuration."""
        # Create temporary Excel file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xlsx', delete=False) as f:
            excel_path = f.name
        
        try:
            data = pd.DataFrame({
                'file path': ['/data/test/file.mzML'],
                'test_col': ['test_value'],
            })
            data.to_excel(excel_path, index=False)
            
            config = {
                'use_search_data': True,
                'search_data_path': excel_path,
            }
            manager = create_search_data_manager(config)
            
            assert manager is not None
            assert manager.is_loaded
            assert len(manager._lookup) == 1
        finally:
            os.unlink(excel_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])






