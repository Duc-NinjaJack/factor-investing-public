# Market Intelligence Module - Dependency Diagram

## Module Dependencies

```mermaid
graph TD
    A[daily_alpha_pulse.py] --> B[data_loader.py]
    A --> C[config.py]
    A --> D[plotly]
    A --> E[pandas]
    A --> F[numpy]
    
    G[simple_daily_pulse.py] --> H[mysql-connector]
    G --> I[yaml]
    G --> E
    G --> F
    
    J[terminal_daily_pulse.py] --> H
    J --> I
    J --> E
    J --> F
    
    B --> C
    B --> H
    B --> E
    B --> F
    
    C --> I
    
    K[database.yml] --> C
    K --> G
    K --> J
```

## External Dependencies

```mermaid
graph TD
    A[Market Intelligence Module] --> B[Python Packages]
    A --> C[MySQL Database]
    A --> D[Configuration Files]
    A --> E[File System]
    
    B --> F[pandas]
    B --> G[numpy]
    B --> H[mysql-connector]
    B --> I[yaml]
    B --> J[plotly]
    B --> K[Standard Library]
    
    K --> L[datetime]
    K --> M[logging]
    K --> N[json]
    K --> O[pathlib]
    K --> P[sys]
    K --> Q[typing]
    K --> R[argparse]
    
    C --> S[equity_history]
    C --> T[vcsc_daily_data_complete]
    C --> U[fundamental_values]
    C --> V[factor_scores_qvm]
    C --> W[vcsc_foreign_flow_summary]
    C --> X[master_info]
    C --> Y[etf_history]
    
    D --> Z[database.yml]
    
    E --> AA[reports/]
    E --> BB[templates/]
```

## Data Flow Dependencies

```mermaid
graph LR
    A[Database Tables] --> B[data_loader.py]
    B --> C[daily_alpha_pulse.py]
    B --> D[simple_daily_pulse.py]
    B --> E[terminal_daily_pulse.py]
    
    F[database.yml] --> G[config.py]
    G --> B
    G --> C
    
    C --> H[HTML Reports]
    C --> I[PDF Reports]
    D --> J[Simple Output]
    E --> K[Terminal Output]
```

## Runtime Dependencies

```mermaid
graph TD
    A[Python Environment] --> B[Package Installation]
    A --> C[MySQL Server]
    A --> D[File Permissions]
    
    B --> E[pip install requirements.txt]
    C --> F[Database Connection]
    D --> G[Write Access to reports/]
    
    F --> H[Table Schema Validation]
    G --> I[Report Generation]
    
    H --> J[Data Availability Check]
    I --> K[Output Files]
```

## Error Handling Dependencies

```mermaid
graph TD
    A[Error Condition] --> B{Error Type}
    
    B -->|Database Connection| C[Connection Retry]
    B -->|Missing Data| D[Graceful Degradation]
    B -->|File System| E[Directory Creation]
    B -->|Configuration| F[Default Values]
    
    C --> G[Log Error]
    D --> H[Use Fallback Data]
    E --> I[Create Missing Directories]
    F --> J[Use Hardcoded Defaults]
    
    G --> K[Continue Execution]
    H --> K
    I --> K
    J --> K
```

## Security Dependencies

```mermaid
graph TD
    A[Security Layer] --> B[Database Security]
    A --> C[File System Security]
    A --> D[Input Validation]
    
    B --> E[Read-only Access]
    B --> F[Connection Management]
    B --> G[SQL Injection Protection]
    
    C --> H[Path Validation]
    C --> I[Safe Directory Creation]
    
    D --> J[Parameterized Queries]
    D --> K[Data Type Validation]
```

## Performance Dependencies

```mermaid
graph TD
    A[Performance Optimization] --> B[Database Performance]
    A --> C[Memory Management]
    A --> D[Query Optimization]
    
    B --> E[Indexed Queries]
    B --> F[Connection Pooling]
    
    C --> G[Pandas DataFrames]
    C --> H[Context Managers]
    C --> I[Lazy Loading]
    
    D --> J[Efficient Joins]
    D --> K[Query Caching]
```

## Deployment Dependencies

```mermaid
graph TD
    A[Deployment] --> B[Environment Setup]
    A --> C[Configuration]
    A --> D[Database Setup]
    
    B --> E[Python 3.x]
    B --> F[Package Installation]
    B --> G[MySQL Server]
    
    C --> H[database.yml]
    C --> I[Directory Structure]
    
    D --> J[Schema Creation]
    D --> K[Data Population]
    D --> L[User Permissions]
    
    E --> M[Runtime Environment]
    F --> M
    G --> M
    H --> M
    I --> M
    J --> M
    K --> M
    L --> M
```
