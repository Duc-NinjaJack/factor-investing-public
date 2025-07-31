# Project Rules and Development Practices

**Project:** Factor Investing Framework - Vietnam Market  
**Version:** 1.0  
**Last Updated:** January 2025  
**Purpose:** Document established patterns, conventions, and best practices  

---

## 📁 Project Structure and Organization

### **Directory Structure**
```
factor-investing-public/
├── docs/                           # Comprehensive documentation
│   ├── 1_investment_methodology/   # Investment philosophy and factor definitions
│   ├── 2_technical_implementation/ # Technical specifications and architecture
│   ├── 3_operational_framework/    # Operational procedures and playbooks
│   └── 4_backtesting_and_research/ # Backtesting methodologies and results
├── production/                     # Production code
│   ├── engine/                     # Core QVM calculation engine
│   ├── scripts/                    # Production execution scripts
│   └── tests/                      # Testing and validation suites
├── scripts/                        # Utility and workflow scripts
│   ├── intermediaries/             # Data processing scripts
│   └── sector_views/               # Sector analysis tools
└── config/                         # Configuration files
```

### **File Naming Conventions**
- **Python files:** `snake_case.py` (e.g., `qvm_engine_v2_enhanced.py`)
- **Configuration files:** `snake_case.yml` or `snake_case.ini`
- **Documentation:** `##_descriptive_name.md` (numbered sections)
- **Test files:** `##_test_description.ipynb` or `##_test_description.md`
- **Backup files:** `original_name_backup_description.ext`

### **Notebook Generation Workflow**
- **Step 1:** Create the markdown (`.md`) file first with all code blocks and documentation
- **Step 2:** Convert the markdown file to Jupyter notebook (`.ipynb`) using `jupytext`
- **Step 3:** Verify the notebook structure and cell execution
- **Rationale:** Markdown files are easier to edit, version control, and maintain than notebook files
- **Command:** `jupytext --to notebook filename.md`

---

## 🐍 Python Code Standards

### **File Headers and Documentation**
```python
"""
Vietnam Factor Investing Platform - Component Name
=================================================
Component: Brief description
Purpose: Detailed purpose and role
Author: Author Name, Title
Date Created: YYYY-MM-DD
Status: PRODUCTION/EXPERIMENTAL/ARCHIVE

Key Features:
1. Feature 1 description
2. Feature 2 description
3. Feature 3 description

Data Sources:
- database_table_name (description)
- another_table (description)

Dependencies:
- pandas >= 1.3.0
- numpy >= 1.21.0
- sqlalchemy >= 1.4.0
"""
```

### **Class Documentation**
```python
class ClassName:
    """
    Brief description of the class.
    
    Attributes:
        attr1 (type): Description
        attr2 (type): Description
    
    Methods:
        method1: Description
        method2: Description
    """
    
    def __init__(self, param1: type, param2: type = default):
        """
        Initialize the class.
        
        Args:
            param1 (type): Description
            param2 (type, optional): Description. Defaults to default.
        """
```

### **Function Documentation**
```python
def function_name(param1: type, param2: type = default) -> return_type:
    """
    Brief description of the function.
    
    Args:
        param1 (type): Description
        param2 (type, optional): Description. Defaults to default.
    
    Returns:
        return_type: Description of return value
    
    Raises:
        ExceptionType: Description of when this exception is raised
    """
```

### **Import Organization**
```python
# Standard library imports
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

# Third-party imports
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import yaml

# Local imports (if any)
from .utils import helper_function
```

### **Variable and Function Naming**
- **Variables:** `snake_case` (e.g., `market_cap`, `factor_scores`)
- **Functions:** `snake_case` (e.g., `calculate_qvm_composite`, `get_market_data`)
- **Classes:** `PascalCase` (e.g., `QVMEngineV2Enhanced`, `EnhancedEVCalculator`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `MAX_LOOKBACK_DAYS`, `DEFAULT_WEIGHTS`)
- **Private methods:** `_leading_underscore` (e.g., `_setup_logging`, `_load_configurations`)

---

## ⚙️ Configuration Management

### **YAML Configuration Structure**
```yaml
# Configuration file header
# Component: Brief description
# Author: Author Name
# Date: YYYY-MM-DD
# Purpose: Detailed purpose

# Main configuration section
main_section:
  # Subsections with clear descriptions
  subsection1:
    parameter1: value1
    parameter2: value2
    
  subsection2:
    nested_param:
      key1: value1
      key2: value2
```

### **Configuration File Naming**
- **Strategy config:** `strategy_config.yml`
- **Database config:** `database.yml`
- **Factor metadata:** `factor_metadata.yml`
- **Sector-specific:** `sector_name_factor_config.yml`

### **Configuration Best Practices**
1. **Centralized configuration** - All parameters in config files
2. **Environment-specific configs** - Separate dev/prod configs
3. **Version control** - Config files in git (exclude credentials)
4. **Documentation** - Clear comments for all parameters
5. **Validation** - Type checking and validation for config values

---

## 🗄️ Database and Data Management

### **Database Connection Pattern**
```python
def _create_database_engine(self):
    """Create database engine with proper error handling."""
    try:
        config = self.config['database']
        connection_string = (
            f"mysql+pymysql://{config['user']}:{config['password']}"
            f"@{config['host']}:{config['port']}/{config['database']}"
        )
        return create_engine(connection_string, pool_recycle=3600)
    except Exception as e:
        self.logger.error(f"Database connection failed: {e}")
        raise
```

### **Data Query Patterns**
```python
def get_data_with_validation(self, query: str, params: dict = None) -> pd.DataFrame:
    """
    Execute query with proper error handling and validation.
    
    Args:
        query (str): SQL query
        params (dict, optional): Query parameters
    
    Returns:
        pd.DataFrame: Query results
    
    Raises:
        DatabaseError: If query fails
    """
    try:
        with self.engine.connect() as conn:
            result = pd.read_sql(query, conn, params=params)
            self.logger.info(f"Retrieved {len(result)} records")
            return result
    except Exception as e:
        self.logger.error(f"Query failed: {e}")
        raise
```

### **Data Quality Checks**
- **Null value handling** - Explicit handling of missing data
- **Data type validation** - Ensure correct data types
- **Range validation** - Check for reasonable value ranges
- **Consistency checks** - Verify data consistency across tables

---

## 📊 Analysis and Testing Patterns

### **Jupyter Notebook Structure**
```python
# ============================================================================
# CELL 1: SETUP AND IMPORTS
# ============================================================================

# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Project-specific imports
from production.engine.qvm_engine_v2_enhanced import QVMEngineV2Enhanced

# ============================================================================
# CELL 2: DATA LOADING
# ============================================================================

# Load data with proper error handling
try:
    data = load_data()
    print(f"✅ Loaded {len(data)} records")
except Exception as e:
    print(f"❌ Data loading failed: {e}")
    raise

# ============================================================================
# CELL 3: ANALYSIS
# ============================================================================

# Perform analysis with clear documentation
print("🔍 Performing analysis...")
results = perform_analysis(data)
print(f"✅ Analysis complete: {len(results)} results")

# ============================================================================
# CELL 4: VISUALIZATION
# ============================================================================

# Create visualizations with proper styling
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# ... visualization code ...
plt.savefig('analysis_results.png', dpi=300, bbox_inches='tight')
plt.show()
```

### **Testing Patterns**
- **Unit tests** - Test individual functions and methods
- **Integration tests** - Test component interactions
- **Validation tests** - Verify data quality and consistency
- **Performance tests** - Test system performance under load

---

## 📝 Documentation Standards

### **Markdown Documentation Structure**
```markdown
# Document Title

**Project:** Component Name  
**Date:** YYYY-MM-DD  
**Purpose:** Brief description  
**Status:** DRAFT/PRODUCTION/ARCHIVE  

---

## 🎯 Executive Summary

Brief overview of the document content.

---

## 📊 Main Content

### **Section 1: Description**
Content with proper formatting.

### **Section 2: Analysis**
- Bullet points for lists
- **Bold text** for emphasis
- `Code snippets` for technical content

---

## 📋 Conclusion

Summary of key findings or next steps.

---

**Document Version:** 1.0  
**Last Updated:** YYYY-MM-DD  
**Next Review:** YYYY-MM-DD
```

### **Documentation Types**
1. **Technical specifications** - Detailed technical documentation
2. **User guides** - How-to documentation for end users
3. **API documentation** - Function and class documentation
4. **Architecture documents** - System design and architecture
5. **Research reports** - Analysis results and findings

---

## 🔄 Version Control and Git Practices

### **Branch Naming**
- **Feature branches:** `feature/description` (e.g., `feature/liquidity-filter`)
- **Bug fixes:** `fix/description` (e.g., `fix/data-validation`)
- **Hotfixes:** `hotfix/description` (e.g., `hotfix/critical-bug`)
- **Analysis branches:** `analysis/description` (e.g., `analysis/liquidity-buckets`)

### **Commit Message Format**
```
Type: Brief description

Detailed description of changes made.

- Bullet point of specific change
- Another specific change
- Impact or reasoning for changes

Files changed:
- file1.py: Description of changes
- file2.yml: Description of changes
```

### **Commit Types**
- **feat:** New feature or enhancement
- **fix:** Bug fix
- **docs:** Documentation changes
- **style:** Code style changes (formatting, etc.)
- **refactor:** Code refactoring
- **test:** Adding or updating tests
- **chore:** Maintenance tasks

### **Git Ignore Patterns**
- **Data files:** `*.csv`, `*.pkl`, `*.parquet`
- **Credentials:** `*credentials*`, `*.env`
- **Cache:** `__pycache__/`, `*.pyc`
- **IDE files:** `.vscode/`, `.idea/`
- **OS files:** `.DS_Store`, `Thumbs.db`

---

## 🚀 Deployment and Production Practices

### **Environment Management**
- **Development:** Local development environment
- **Staging:** Pre-production testing environment
- **Production:** Live production environment

### **Configuration Management**
- **Environment variables** for sensitive data
- **Configuration files** for application settings
- **Database credentials** in secure storage
- **API keys** in environment variables

### **Logging Standards**
```python
import logging

# Setup logging with proper configuration
def setup_logging(level: str = 'INFO') -> logging.Logger:
    """Setup logging with standard configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
```

### **Error Handling**
```python
def robust_function(data: pd.DataFrame) -> pd.DataFrame:
    """Function with comprehensive error handling."""
    try:
        # Main logic
        result = process_data(data)
        logger.info("Data processing completed successfully")
        return result
    except ValueError as e:
        logger.error(f"Invalid data format: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
```

---

## 📈 Performance and Optimization

### **Performance Monitoring**
- **Execution time** tracking for long-running operations
- **Memory usage** monitoring for large datasets
- **Database query** performance optimization
- **Caching** for frequently accessed data

### **Code Optimization**
- **Vectorized operations** using pandas/numpy
- **Efficient data structures** for large datasets
- **Lazy evaluation** for expensive computations
- **Parallel processing** for independent operations

---

## 🔒 Security and Data Protection

### **Data Security**
- **Encryption** for sensitive data
- **Access controls** for database connections
- **Audit logging** for data access
- **Data anonymization** for research purposes

### **Code Security**
- **Input validation** for all user inputs
- **SQL injection** prevention
- **Credential management** best practices
- **Regular security** updates

---

## 📋 Quality Assurance

### **Code Review Checklist**
- [ ] **Functionality** - Does the code work as intended?
- [ ] **Performance** - Is the code efficient?
- [ ] **Security** - Are there security vulnerabilities?
- [ ] **Documentation** - Is the code well-documented?
- [ ] **Testing** - Are there appropriate tests?
- [ ] **Standards** - Does the code follow project standards?

### **Testing Requirements**
- **Unit tests** for all new functions
- **Integration tests** for component interactions
- **Regression tests** for existing functionality
- **Performance tests** for critical paths

---

## 🎯 Best Practices Summary

### **General Principles**
1. **Consistency** - Follow established patterns
2. **Documentation** - Document everything
3. **Testing** - Test thoroughly
4. **Security** - Prioritize security
5. **Performance** - Optimize for performance
6. **Maintainability** - Write maintainable code

### **Code Quality**
1. **Readability** - Write clear, readable code
2. **Modularity** - Break code into logical modules
3. **Reusability** - Design for reuse
4. **Error handling** - Handle errors gracefully
5. **Logging** - Log important events
6. **Validation** - Validate inputs and outputs

### **Project Management**
1. **Version control** - Use git effectively
2. **Documentation** - Keep documentation up to date
3. **Testing** - Maintain comprehensive test coverage
4. **Deployment** - Use proper deployment practices
5. **Monitoring** - Monitor system performance
6. **Security** - Maintain security best practices

---

**Document Version:** 1.0  
**Last Updated:** January 2025  
**Next Review:** Quarterly  
**Maintained By:** Development Team