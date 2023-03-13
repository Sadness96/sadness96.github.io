---
title: NPOI 以内存方式导出 Excel
date: 2023-03-10 10:48:00
tags: [c#,helper,npoi,excel]
categories: C#.Net
---
### 服务端以内存方式导出 Excel 并设置自动列宽
<!-- more -->
### 简介
基于 [NPOI/DOCX 帮助类](https://sadness96.github.io/blog/2017/05/27/csharp-NPOIHelper/) 修改，在服务端导出 Excel 应尽量直接在内存中处理，减少磁盘写入，同时实现设置自动列宽。

### 代码
#### 帮助类修改
基于帮助类中 FillDataTable 方法修改
``` csharp
/// <summary>
/// 设置自动列宽
/// </summary>
/// <param name="sheet"></param>
/// <param name="cols"></param>
public static void AutoColumnWidth(ISheet sheet, int cols)
{
    for (int col = 0; col <= cols; col++)
    {
        //获取当前列宽度
        int columnWidth = sheet.GetColumnWidth(col) / 256;
        for (int rowIndex = 0; rowIndex <= sheet.LastRowNum; rowIndex++)
        {
            IRow row = sheet.GetRow(rowIndex);
            ICell cell = row.GetCell(col);
            //获取当前单元格的内容宽度
            int contextLength = Encoding.UTF8.GetBytes(cell.ToString()).Length;
            columnWidth = columnWidth < contextLength ? contextLength : columnWidth;

        }
        sheet.SetColumnWidth(col, columnWidth * 256);
    }
}

/// <summary>
/// 在指定Excel中指定Sheet指定位置填充DataTable
/// 在内存中处理
/// </summary>
/// <param name="strSheetName">需要填充的Sheet名称(如果没有则添加,如果冲突则使用冲突Sheet)</param>
/// <param name="dtSourceData">DataTable数据</param>
/// <param name="WhetherThereFieldName">是否有列名(true保留DataTable字段名)</param>
/// <param name="iRows">起始行</param>
/// <param name="iColumn">起始列</param>
/// <returns>成功返回true,失败返回false</returns>
public static MemoryStream FillDataTableStream(string strSheetName, DataTable dtSourceData, bool WhetherThereFieldName, int iRows, int iColumn)
{
    try
    {
        if (string.IsNullOrEmpty(strSheetName) || dtSourceData.Rows.Count < 1)
        {
            return null;
        }

        IWorkbook iWorkBook = new XSSFWorkbook();
        ISheet iSheet = iWorkBook.CreateSheet(strSheetName);

        if (WhetherThereFieldName)
        {
            IRow rowDataTableField = iSheet.CreateRow(iRows);
            for (int iDataTableColumns = 0; iDataTableColumns < dtSourceData.Columns.Count; iDataTableColumns++)
            {
                ICell cellErrstatist = rowDataTableField.CreateCell(iDataTableColumns + iColumn);
                cellErrstatist.SetCellValue(dtSourceData.Columns[iDataTableColumns].ColumnName);
            }
            for (int iDataTableRows = 0; iDataTableRows < dtSourceData.Rows.Count; iDataTableRows++)
            {
                IRow rowDataTable = iSheet.CreateRow(iDataTableRows + iRows + 1);
                for (int iDataTableColumns = 0; iDataTableColumns < dtSourceData.Columns.Count; iDataTableColumns++)
                {
                    ICell cellErrstatist = rowDataTable.CreateCell(iDataTableColumns + iColumn);
                    string strSourceData = dtSourceData.Rows[iDataTableRows][iDataTableColumns].ToString();
                    Regex regexIsNumeric = new Regex(@"^(-?\d+)(\.\d+)?$");
                    if (regexIsNumeric.IsMatch(strSourceData))
                    {
                        cellErrstatist.SetCellValue(double.Parse(strSourceData));
                    }
                    else
                    {
                        cellErrstatist.SetCellValue(strSourceData);
                    }
                }

                AutoColumnWidth(iSheet, dtSourceData.Columns.Count - 1);
            }
        }
        else
        {
            for (int iDataTableRows = 0; iDataTableRows < dtSourceData.Rows.Count; iDataTableRows++)
            {
                IRow rowDataTable = iSheet.CreateRow(iDataTableRows + iRows);
                for (int iDataTableColumns = 0; iDataTableColumns < dtSourceData.Columns.Count; iDataTableColumns++)
                {
                    ICell cellErrstatist = rowDataTable.CreateCell(iDataTableColumns + iColumn);
                    string strSourceData = dtSourceData.Rows[iDataTableRows][iDataTableColumns].ToString();
                    Regex regexIsNumeric = new Regex(@"^(-?\d+)(\.\d+)?$");
                    if (regexIsNumeric.IsMatch(strSourceData))
                    {
                        cellErrstatist.SetCellValue(double.Parse(strSourceData));
                    }
                    else
                    {
                        cellErrstatist.SetCellValue(strSourceData);
                    }
                }
            }
        }

        MemoryStream stream = new MemoryStream();
        iWorkBook.Write(stream, false);
        iWorkBook.Close();
        return stream;
    }
    catch (Exception ex)
    {
        return null;
    }
}
```

#### 代码调用
调用使用服务端 .NET 6.0 API
``` csharp
[HttpGet]
public IActionResult Export()
{
    DataTable dataTable = new DataTable();

    ...

    MemoryStream msExcel = Helper.ExcelHelper.FillDataTableStream("Sheet1", dataTable, true, 0, 0);
    MemoryStream memoryStream = new MemoryStream(msExcel.ToArray());
    return File(memoryStream, "application/octet-stream", Path.GetFileName($"{Guid.NewGuid()}.xlsx"), true);
}
```