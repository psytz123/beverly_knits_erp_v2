Phase 1: Demand Consolidation

- Sales Orders (priority 1)
- Consistent pattern forecasts (priority 2)
  ↓
  Phase 2: Inventory Assessment (Multi-level)
- F01 (immediate availability)
- I01 (2 days - pending QC)
- G00/G02 (30 days - in process)
- Active KO balance (45 days)
  ↓
  Phase 3: Net Requirements Calculation
- Net = Demand - All Available Inventory
- Time-phased netting by availability date
  ↓
  Phase 4: BOM Explosion (Net requirements only)
- Only explode what needs to be produced
- Calculate yarn requirements
  ↓
  Phase 5: Procurement & Production Planning
- Generate KO recommendations
- Yarn procurement plan with priorities
- Allocation status (Hard/Soft/Available)
  ↓
  Phase 6: Optimization & Output
- Consolidated recommendations
- Management reports

# End to End forecasting

## ML/AI tab

historical sales + forecast + sales orders api/sales-order/plan/list = forecasted future demand

* input: /api/sales-order/plan/list/ & /api/report/sales_activity

### **
    Production tab** 

1. Future demand - total current inventory = required fabric production for future demand

   Input

   * `GET /api/greige/g00` - Greige stage 1 inventory
   * `GET /api/greige/g02` - Greige stage 2 inventory
   * `GET /api/finished/i01` - QC/Inspection inventory
   * `GET /api/finished/f01` - Finished goods inventory
   * `GET /api/knitorder/list` - Fetch knit orders `

   output: Forecasted Production Orders (Net Requirements) & Forecasted Fabric Requirements (90-Day Projection)
2. req. fabric production X bom = req yarn demand for required fabric production for future demand

   ## inventory tab
3. required yarn demand - current unallocated invnetory = forecasted yarn consumption for req. yarn demand for required fabric production for future demand

   * `GET /api/yarn/active `

   OUTPUT: Current Yarn Shortages Affecting Active Orders
4. efab yarn demand + forecasted yarn consumption for req. yarn demand for required fabric production for future demand = total yarn demand by week

   - `GET /api/report/yarn_demand` - Standard yarn demand

    OUTPUT Time-Phased Yarn Purchase Order & Forecasted Yarn Shortages (90-Day Projection)

<h2>Sales Activity Report</h2>
  <div class='grid-x'>
    <div class='cell small-3'>
      <h3>Start Date:</h3><br><div id="startDate"></div>
    </div>
    <div class='cell small-1'></div>
    <div class='cell small-3'>
      <h3>End Date:</h3><br><div id="endDate"></div>
    </div>
    <div class='cell small-3'>
      <button type="submit" id="runReport" onClick='runReport()' class='button'>Run The Report 
      <i id="reportSpinner" class="fa fa-spinner fa-spin"></i></button>
    </div>
  </div>
  <hr>
  <div id="detailGrid"></div>
</div>

<script>
var reportStore = [];

$(document).ready(function() {
  $("#reportSpinner").hide();
  //runReport();
});

var endingDate = new Date;
endingDate.setDate(endingDate.getDate() - 1);

var startDate = $('#startDate').dxDateBox({
  name: 'start_date',
  width: 150,
  max: endingDate,
  min: new Date("01-01-2020"),
  //value: new Date("09-01-2023"),
}).dxDateBox('instance');

var endDate = $('#endDate').dxDateBox({
  name: 'end_date',
  width: 150,
  max: endingDate,
  min: new Date("01-01-2020"),
  //value: new Date("09-07-2023"),
}).dxDateBox('instance');

var detailGrid = $('#detailGrid').dxDataGrid({
  datasource: new DevExpress.data.ArrayStore({
    key: 'id',
    data: reportStore
  }),
  allowColumnReordering: true,
  allowColumnResizing: false,
  columnAutoWidth: true,
  columnChooser: { enabled: false },
  columnFixing: {
    enabled: true
  },
  filterRow: { visible: true },
  paging: { enabled: true },
  height: $(window).height() - 160,
  width: $('.mainContentWrapper').width() - 200,
  "export": {
    enabled: true,
    fileName: 'Sales Activity Report',
  },
  grouping: {
    autoExpandAll: false,
  },
  groupPanel: {
    visible: true,
  },
  columns: [{
      dataField: 'serial_number',
      dataType: 'string',
      caption:   'Document',
    }, {
      dataField: 'shipping_order.invoiced_date',
      dataType:  'date',
      caption:   'Invoice Date',
    }, {
      dataField: 'bill_to.name',
      dataType: 'string',
      caption: 'Customer',
    }, {
      dataField: 'style',
      dataType: 'string',
      caption: 'Style',
      calculateCellValue: function(rowData) {
        if (rowData.cf_version) {
           return rowData.cf_version.code;
        } else {
           return "(Greige) " + rowData.knit_version.knit_style_base.base_style + '/' + rowData.knit_version.version;
        }
      },
    }, {
      dataField: "shipped",
      dataType: 'number',
      caption: "Qty Shipped",
      format: {
        precision: 1,
      },
      calculateCellValue: function(rowData) {
        if (rowData.use_metric) {
          return +rowData.shipped_metric;
        }else {
          return +rowData.shipped;
        }
      }
    }, {
      dataField: 'qty_uom',
      dataType: 'string',
      caption: 'UOM',
    }, {
      dataField: "unit_price",
      dataType: 'number',
      caption: 'Unit Price',
      alignment: 'right',
      format: {
        type:   'currency',
        precision: 2,
      },
      calculateCellValue: function(rowData) {
        if (rowData.use_metric) {
          return +rowData.unit_price_metric;
        }else {
          return +rowData.unit_price;
        }
      }
    }, {
      dataField: "line_price",
      dataType: 'number',
      caption: 'Line Price',
      alignment: 'right',
      format: {
        type:   'currency',
        precision: 2,
      },
      calculateCellValue: function(rowData) {
        if (rowData.use_metric) {
          return +rowData.line_price_metric;
        }else {
          return +rowData.line_price;
        }
      }
    }, {
      dataField: "shipping_order.invoices[0].serial_number",
      dataType: 'string',
      caption: '1st Invoice',
    }, {
      dataField: "shipping_order.sales_rep.name",
      dataType: 'string',
      caption: 'Sales Rep',
    }, {
      dataField: 'shipping_order.sales_rep.commission',
      dataType: 'string',
      caption: 'Commission',
    }, {
      dataField: "bill_to.invoice_customer",
      dataType: 'string',
      caption: 'Selling Co',
    }, {
      dataField: "effective_purchase_order",
      dataType: 'string',
      caption: 'PO#',
    }
  ],
  summary: {
    groupItems: [
      {
        column: "style",
        summaryType: "count",
        valueFormat: "fixedPoint",
        showInGroupFooter: false,
        displayFormat: "{0} orders",
      }, {
        column: "shipped",
        summaryType: "sum",
        //valueFormat: "fixedPoint",
        valueFormat: { precision: 2 },
        showInGroupFooter: false,
        displayFormat: "{0} lbs",
      }
    ],
  },
}).dxDataGrid("instance");

function runReport() {
  $("#reportSpinner").show();
  var data = new Object;
  data.startDate = startDate.option('value').toISOString();
  data.endDate   = endDate.option('value').toISOString();
  $.ajax( '/api/report/sales_activity', {
    method: "POST",
    data: data,
    cache: false,
    xhrFields: { withCredientials: true }
  })
  .done(function(d1) {
    var data=JSON.parse(d1);
    var newDetailData = new DevExpress.data.ArrayStore({
      key: 'id',
      data: data
    })
    detailGrid.option('dataSource', newDetailData);
    detailGrid.refresh();
    $("#reportSpinner").hide();
  })
  .fail(function(xhr) {
    console.log(xhr);
    $("#reportSpinner").hide();
  })
  return;
}
</script>

</div>
    </div>


    

</div>

<script>$(document).foundation();</script>

</body>
</html>

<script type="text/javascript">
    $( document ).ready(function() {
    
    
  
  
        var fullURL = window.location;
        var baseURL = fullURL.protocol + "//" + fullURL.host;
        var timer = setInterval(
            function() {
        
            },
            60000
        );

        function finishJob( job_id, job_type ) {
            clearInterval(timer);

            var actionURL;
            switch( job_type ) {
            
                case 'yarn_reconcile_import':
                    actionURL = '/admin/reconcile_results/' + job_id;
                    break;
            
                default:
                    return;
            }

            var finishURL = baseURL + actionURL;
            $.ajax({
                url: finishURL,
                type: 'GET',
                success: function( data ) {
                    if( data.error === 1 ) {
                        $.growl({ message: data.message, style: "error", title: "Error!", fixed: true });
                    } else {
                        $.growl({ message: data.message, style: "notice", title: "Success!", fixed: true });
                    }
                    console.log( "Finished job " + job_id + "." );
                },
                error: function(event, stat, error) {
                    var str = 'JSON request failed. Status: ' + stat + ' Error: ' + error;
                    console.log(str);
                },
            });
        }
  

    });
</script>
