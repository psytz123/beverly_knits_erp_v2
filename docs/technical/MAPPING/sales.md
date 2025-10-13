Request URL

https://efab.bkiapps.com/api/report/sales_activity

Request Method

POST

Status Code

200 OK

Remote Address

52.22.59.254:443

Referrer Policy

strict-origin-when-cross-origin

POST /api/report/sales_activity HTTP/1.1
Accept: */*
Accept-Encoding: gzip, deflate, br, zstd
Accept-Language: en-US,en;q=0.9
Connection: keep-alive
Content-Length: 75
Content-Type: application/x-www-form-urlencoded; charset=UTF-8
Cookie: dancer.session=aNDE57iQnMQ6fBlOO9KONL06oEdhlsvb
Host: efab.bkiapps.com
Origin: https://efab.bkiapps.com
Referer: https://efab.bkiapps.com/reports/sales_activity
Sec-Fetch-Dest: empty
Sec-Fetch-Mode: cors
Sec-Fetch-Site: same-origin
User-Agent: Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Mobile Safari/537.36
X-Requested-With: XMLHttpRequest
sec-ch-ua: "Chromium";v="140", "Not=A?Brand";v="24", "Google Chrome";v="140"
sec-ch-ua-mobile: ?1
sec-ch-ua-platform: "Android


<div class="mainContentWrapper">
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
      <button type="submit" id="runReport" onClick='runReport()' class='button'>Run The Report&nbsp;
      <i id="reportSpinner" class="fa fa-spinner fa-spin"></i></button>
    </div>
  </div>
  <hr>
  <div id="detailGrid"></div>
</div>

`<script>`
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
`</script>`

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
