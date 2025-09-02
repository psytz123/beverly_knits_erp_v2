import pandas as pd

# Check unique styles in sales vs BOM
sales = pd.read_csv("/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/ERP Data/Sales Activity Report.csv")
bom = pd.read_csv("/mnt/c/finalee/beverly_knits_erp_v2/data/production/5/BOM_updated.csv")

sales_styles = set(sales['fStyle#'].unique())
bom_styles = set(bom['Style#'].unique())

print(f"Sales has {len(sales_styles)} unique styles")
print(f"BOM has {len(bom_styles)} unique styles")

# Check for any overlap
overlap = sales_styles.intersection(bom_styles)
print(f"\nOverlapping styles: {len(overlap)}")

if overlap:
    print("Sample overlapping styles:", list(overlap)[:5])
else:
    print("\nNo overlapping styles found!")
    print("Sample sales styles:", list(sales_styles)[:5])
    print("Sample BOM styles:", list(bom_styles)[:5])
