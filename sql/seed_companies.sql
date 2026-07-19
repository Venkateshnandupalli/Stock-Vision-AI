-- =============================================================================
-- StockVision AI — Seed Companies
-- =============================================================================
-- Run AFTER schema.sql to populate the companies reference table.
-- This data matches the COMPANY_INFO dict in src/utils/config.py.
-- =============================================================================

INSERT INTO companies (ticker, company_name, sector, industry, exchange, currency, is_active)
VALUES
    ('TCS.NS',        'Tata Consultancy Services',    'Information Technology', 'IT Services',        'NSE', 'INR', TRUE),
    ('INFY.NS',       'Infosys Limited',              'Information Technology', 'IT Services',        'NSE', 'INR', TRUE),
    ('WIPRO.NS',      'Wipro Limited',                'Information Technology', 'IT Services',        'NSE', 'INR', TRUE),
    ('HDFCBANK.NS',   'HDFC Bank Limited',            'Banking',               'Private Sector Bank', 'NSE', 'INR', TRUE),
    ('ICICIBANK.NS',  'ICICI Bank Limited',           'Banking',               'Private Sector Bank', 'NSE', 'INR', TRUE),
    ('SBIN.NS',       'State Bank of India',          'Banking',               'Public Sector Bank',  'NSE', 'INR', TRUE),
    ('RELIANCE.NS',   'Reliance Industries Limited',  'Energy',                'Oil & Gas',           'NSE', 'INR', TRUE),
    ('ONGC.NS',       'Oil and Natural Gas Corp',     'Energy',                'Oil & Gas',           'NSE', 'INR', TRUE),
    ('TATAMOTORS.NS', 'Tata Motors Limited',          'Automobile',            'Auto Manufacturer',   'NSE', 'INR', TRUE),
    ('MARUTI.NS',     'Maruti Suzuki India Limited',  'Automobile',            'Auto Manufacturer',   'NSE', 'INR', TRUE),
    ('^NSEI',         'NIFTY 50 Index',               'Benchmark',             'Market Index',        'NSE', 'INR', TRUE)
ON CONFLICT (ticker) DO UPDATE SET
    company_name = EXCLUDED.company_name,
    sector       = EXCLUDED.sector,
    industry     = EXCLUDED.industry,
    is_active    = EXCLUDED.is_active;

-- Confirm seeded rows
SELECT company_id, ticker, company_name, sector, exchange
FROM   companies
ORDER  BY sector, ticker;
