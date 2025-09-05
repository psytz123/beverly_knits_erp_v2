#!/usr/bin/env python3
"""
Create BOM to eFab mapping using nested gBase and base_style fields
"""

import json
import pandas as pd

print('Extracting base_style and version data from nested structures...')
print('='*70)

# Load the complete data
with open('data/production/5/ERP Data/efab_api_styles_complete.json', 'r') as f:
    styles_data = json.load(f)

# Extract all style codes including nested ones
all_style_codes = set()
extracted_mappings = []

for record in styles_data:
    # Direct code field
    if 'code' in record and record['code']:
        all_style_codes.add(str(record['code']).upper())
    
    # Check cf_version nested structure
    if 'cf_version' in record and isinstance(record['cf_version'], dict):
        cf_ver = record['cf_version']
        
        # Extract cf_version code
        if 'code' in cf_ver:
            all_style_codes.add(str(cf_ver['code']).upper())
        
        # Look for base_style in nested f_base
        if 'f_base' in cf_ver and isinstance(cf_ver['f_base'], dict):
            f_base = cf_ver['f_base']
            
            if 'base_style' in f_base:
                base_style = str(f_base['base_style']).upper()
                all_style_codes.add(base_style)
                
                # Store mapping info
                extracted_mappings.append({
                    'api_code': record.get('code', ''),
                    'base_style': f_base['base_style'],
                    'cf_code': cf_ver.get('code', ''),
                    'description': cf_ver.get('description', '')
                })
            
            # Check for knit_style_base
            if 'f_version' in f_base and isinstance(f_base['f_version'], dict):
                f_ver = f_base['f_version']
                if 'knit_style_base' in f_ver and isinstance(f_ver['knit_style_base'], dict):
                    ks_base = f_ver['knit_style_base']
                    if 'base_style' in ks_base:
                        all_style_codes.add(str(ks_base['base_style']).upper())

print(f'Total unique style codes extracted: {len(all_style_codes)}')

# Show sample of extracted codes
print('\nSample extracted style codes:')
sample_codes = list(all_style_codes)[:20]
for code in sample_codes:
    print(f'  {code}')

# Load BOM data
bom = pd.read_csv('data/production/5/ERP Data/BOM_updated.csv')
bom_styles = set(bom['Style#'].dropna().str.strip().str.upper())

print(f'\nTotal BOM styles: {len(bom_styles)}')

# Check matches with all extracted codes
direct_matches = bom_styles & all_style_codes

# Also check without spaces
bom_no_space = set()
for style in bom_styles:
    if isinstance(style, str):
        bom_no_space.add(style.replace(' ', ''))

space_matches = bom_no_space & all_style_codes

# Check base code matches
base_matches = set()
for bom_style in bom_styles:
    base = bom_style.split('/')[0].strip().replace(' ', '')
    if base in all_style_codes:
        base_matches.add(bom_style)

total_matches = direct_matches | space_matches | base_matches

print(f'\nMATCHING RESULTS WITH NESTED DATA:')
print(f'Direct matches: {len(direct_matches)}')
print(f'Space-removed matches: {len(space_matches)}')
print(f'Base code matches: {len(base_matches)}')
print(f'Total unique matches: {len(total_matches)} ({len(total_matches)/len(bom_styles)*100:.1f}%)')

# Show some matched examples
if total_matches:
    print('\nSample matched styles:')
    for match in list(total_matches)[:15]:
        print(f'  {match}')

# Create and save final mapping
print('\nCreating final comprehensive mapping...')
mapping_records = []

for bom_style in bom_styles:
    bom_clean = bom_style.replace(' ', '')
    bom_base = bom_style.split('/')[0].strip().replace(' ', '')
    
    mapped_to = None
    match_type = 'unmapped'
    
    if bom_style in all_style_codes:
        mapped_to = bom_style
        match_type = 'direct'
    elif bom_clean in all_style_codes:
        mapped_to = bom_clean
        match_type = 'space_removed'
    elif bom_base in all_style_codes:
        mapped_to = bom_base
        match_type = 'base_code'
    
    mapping_records.append({
        'bom_style': bom_style,
        'mapped_to': mapped_to,
        'match_type': match_type
    })

mapping_df = pd.DataFrame(mapping_records)
mapping_df.to_csv('data/production/5/ERP Data/bom_efab_gbase_mapping.csv', index=False)

# Calculate statistics
unmapped_df = mapping_df[mapping_df['match_type'] == 'unmapped']
mapped_df = mapping_df[mapping_df['match_type'] != 'unmapped']

print(f'\nFINAL MAPPING STATISTICS:')
print(f'Total BOM styles: {len(mapping_df)}')
print(f'Successfully mapped: {len(mapped_df)} ({len(mapped_df)/len(mapping_df)*100:.1f}%)')
print(f'Unmapped: {len(unmapped_df)} ({len(unmapped_df)/len(mapping_df)*100:.1f}%)')

# Calculate BOM entry coverage
bom_full = pd.read_csv('data/production/5/ERP Data/BOM_updated.csv')
bom_full['style_upper'] = bom_full['Style#'].str.strip().str.upper()
bom_with_mapping = bom_full.merge(
    mapping_df,
    left_on='style_upper',
    right_on='bom_style',
    how='left'
)
entries_mapped = bom_with_mapping[bom_with_mapping['match_type'] != 'unmapped'].dropna(subset=['match_type']).shape[0]
print(f'\nBOM entries covered: {entries_mapped:,} out of {len(bom_full):,} ({entries_mapped/len(bom_full)*100:.1f}%)')

print(f'\nMapping saved to: bom_efab_gbase_mapping.csv')