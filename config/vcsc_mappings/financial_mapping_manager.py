#!/usr/bin/env python3
"""
Financial Mapping Manager
=========================

This module provides a class to manage financial statement mappings
between VCSC codes and database item_ids dynamically.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

class FinancialMappingManager:
    """
    Manages financial statement mappings between VCSC codes and database item_ids.
    
    This class loads mapping configurations from JSON files and provides
    methods to get the correct database item_ids based on sector and financial item type.
    """
    
    def __init__(self, mappings_dir: Optional[str] = None):
        """
        Initialize the mapping manager.
        
        Args:
            mappings_dir: Directory containing mapping JSON files. 
                         If None, uses default production/database/mappings directory.
        """
        if mappings_dir is None:
            # Find the project root and use default mappings directory
            current_path = Path.cwd()
            while not (current_path / 'production').is_dir():
                if current_path.parent == current_path:
                    raise FileNotFoundError("Could not find the 'production' directory.")
                current_path = current_path.parent
            
            mappings_dir = current_path / 'production' / 'database' / 'mappings'
        
        self.mappings_dir = Path(mappings_dir)
        self.corp_mappings = None
        self.bank_mappings = None
        self._load_mappings()
    
    def _load_mappings(self):
        """Load mapping configurations from JSON files."""
        try:
            # Load corporate mappings
            corp_file = self.mappings_dir / 'corp_code_name_mapping.json'
            with open(corp_file, 'r', encoding='utf-8') as f:
                self.corp_mappings = json.load(f)
            
            # Load bank mappings
            bank_file = self.mappings_dir / 'bank_code_name_mapping.json'
            with open(bank_file, 'r', encoding='utf-8') as f:
                self.bank_mappings = json.load(f)
                
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Mapping file not found: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in mapping file: {e}")
    
    def get_item_id(self, sector: str, item_type: str) -> Tuple[int, str]:
        """
        Get the database item_id and statement_type for a given sector and item type.
        
        Args:
            sector: Company sector (e.g., 'Banks', 'Food & Beverage', etc.)
            item_type: Type of financial item ('net_profit', 'revenue', 'total_assets')
            
        Returns:
            Tuple of (item_id, statement_type)
            
        Raises:
            ValueError: If sector or item_type is not supported
        """
        # Determine if this is a bank based on sector
        is_bank = self._is_bank_sector(sector)
        
        # Select appropriate mapping
        if is_bank:
            mappings = self.bank_mappings.get('database_mappings', {})
        else:
            mappings = self.corp_mappings.get('database_mappings', {})
        
        # Get the mapping for the requested item type
        if item_type not in mappings:
            raise ValueError(f"Unsupported item_type: {item_type}. "
                           f"Supported types: {list(mappings.keys())}")
        
        mapping = mappings[item_type]
        item_id = mapping['item_id']
        statement_type = mapping['statement_type']
        
        return item_id, statement_type
    
    def get_net_profit_mapping(self, sector: str) -> Tuple[int, str]:
        """Get NetProfit item_id and statement_type for the given sector."""
        return self.get_item_id(sector, 'net_profit')
    
    def get_revenue_mapping(self, sector: str) -> Tuple[int, str]:
        """Get Revenue item_id and statement_type for the given sector."""
        return self.get_item_id(sector, 'revenue')
    
    def get_total_assets_mapping(self, sector: str) -> Tuple[int, str]:
        """Get TotalAssets item_id and statement_type for the given sector."""
        return self.get_item_id(sector, 'total_assets')
    
    def get_all_mappings(self, sector: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all available mappings for a given sector.
        
        Args:
            sector: Company sector
            
        Returns:
            Dictionary containing all mappings for the sector
        """
        is_bank = self._is_bank_sector(sector)
        
        if is_bank:
            return self.bank_mappings.get('database_mappings', {})
        else:
            return self.corp_mappings.get('database_mappings', {})
    
    def _is_bank_sector(self, sector: str) -> bool:
        """
        Determine if a sector represents a bank.
        
        Args:
            sector: Company sector string
            
        Returns:
            True if the sector represents a bank, False otherwise
        """
        if not sector:
            return False
        
        sector_lower = sector.lower()
        bank_keywords = ['bank', 'credit', 'financial institution']
        
        return any(keyword in sector_lower for keyword in bank_keywords)
    
    def get_vcsc_code(self, sector: str, item_type: str) -> str:
        """
        Get the VCSC code for a given sector and item type.
        
        Args:
            sector: Company sector
            item_type: Type of financial item
            
        Returns:
            VCSC code string
        """
        is_bank = self._is_bank_sector(sector)
        
        if is_bank:
            mappings = self.bank_mappings.get('database_mappings', {})
        else:
            mappings = self.corp_mappings.get('database_mappings', {})
        
        if item_type not in mappings:
            raise ValueError(f"Unsupported item_type: {item_type}")
        
        return mappings[item_type]['vcsc_code']
    
    def get_description(self, sector: str, item_type: str) -> str:
        """
        Get the description for a given sector and item type.
        
        Args:
            sector: Company sector
            item_type: Type of financial item
            
        Returns:
            Description string
        """
        is_bank = self._is_bank_sector(sector)
        
        if is_bank:
            mappings = self.bank_mappings.get('database_mappings', {})
        else:
            mappings = self.corp_mappings.get('database_mappings', {})
        
        if item_type not in mappings:
            raise ValueError(f"Unsupported item_type: {item_type}")
        
        return mappings[item_type]['description']
    
    def validate_mappings(self) -> Dict[str, Any]:
        """
        Validate that all required mappings are present and consistent.
        
        Returns:
            Dictionary with validation results
        """
        required_items = ['net_profit', 'revenue', 'total_assets']
        results = {
            'corporate_mappings': {},
            'bank_mappings': {},
            'errors': [],
            'warnings': []
        }
        
        # Check corporate mappings
        corp_mappings = self.corp_mappings.get('database_mappings', {})
        for item in required_items:
            if item in corp_mappings:
                mapping = corp_mappings[item]
                results['corporate_mappings'][item] = {
                    'vcsc_code': mapping.get('vcsc_code'),
                    'item_id': mapping.get('item_id'),
                    'statement_type': mapping.get('statement_type'),
                    'description': mapping.get('description')
                }
            else:
                results['errors'].append(f"Missing corporate mapping for {item}")
        
        # Check bank mappings
        bank_mappings = self.bank_mappings.get('database_mappings', {})
        for item in required_items:
            if item in bank_mappings:
                mapping = bank_mappings[item]
                results['bank_mappings'][item] = {
                    'vcsc_code': mapping.get('vcsc_code'),
                    'item_id': mapping.get('item_id'),
                    'statement_type': mapping.get('statement_type'),
                    'description': mapping.get('description')
                }
            else:
                results['errors'].append(f"Missing bank mapping for {item}")
        
        return results
    
    def print_mappings_summary(self):
        """Print a summary of all available mappings."""
        print("📋 Financial Statement Mappings Summary")
        print("=" * 60)
        
        print("\n🏢 Corporate Mappings:")
        print("-" * 30)
        corp_mappings = self.corp_mappings.get('database_mappings', {})
        for item_type, mapping in corp_mappings.items():
            print(f"  {item_type}:")
            print(f"    VCSC Code: {mapping['vcsc_code']}")
            print(f"    Item ID: {mapping['item_id']}")
            print(f"    Statement Type: {mapping['statement_type']}")
            print(f"    Description: {mapping['description']}")
            print()
        
        print("\n🏦 Bank Mappings:")
        print("-" * 30)
        bank_mappings = self.bank_mappings.get('database_mappings', {})
        for item_type, mapping in bank_mappings.items():
            print(f"  {item_type}:")
            print(f"    VCSC Code: {mapping['vcsc_code']}")
            print(f"    Item ID: {mapping['item_id']}")
            print(f"    Statement Type: {mapping['statement_type']}")
            print(f"    Description: {mapping['description']}")
            print()


# Convenience function for quick access
def get_financial_mapping_manager() -> FinancialMappingManager:
    """
    Get a FinancialMappingManager instance with default configuration.
    
    Returns:
        FinancialMappingManager instance
    """
    return FinancialMappingManager()


if __name__ == "__main__":
    # Test the mapping manager
    manager = FinancialMappingManager()
    manager.print_mappings_summary()
    
    # Test validation
    validation_results = manager.validate_mappings()
    print("\n🔍 Validation Results:")
    print(f"Errors: {validation_results['errors']}")
    print(f"Warnings: {validation_results['warnings']}")
    
    # Test getting mappings
    print("\n🧪 Test Mappings:")
    print(f"VCB (Bank) NetProfit: {manager.get_net_profit_mapping('Banks')}")
    print(f"VNM (Non-Bank) Revenue: {manager.get_revenue_mapping('Food & Beverage')}")
    print(f"VCB (Bank) TotalAssets: {manager.get_total_assets_mapping('Banks')}") 