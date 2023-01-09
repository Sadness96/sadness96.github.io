---
title: NPOI/DOCX 帮助类
date: 2017-05-27 14:30:00
tags: [c#,helper,npoi,docx,excel,word]
categories: C#.Net
---
### 使用 NPOI/DocX 二次封装Office（Word、Excel）帮助类
<!-- more -->
#### 简介
工作中需要大量生成导出报表或合同证书文件，原理为使用Excel或Word模板，批量替换标签以达到效果。

#### 设计
由于原类库都属于基础方法，二次封装后具有更简易的使用方式，可直接传入生成的数据集或标签替换集合。

#### 引用库介绍
由于微软默认推荐的类库 [Microsoft.Office.Interop.Word](https://docs.microsoft.com/en-us/dotnet/api/microsoft.office.interop.word?redirectedfrom=MSDN&view=word-pia) 与 [Microsoft.Office.Interop.Excel](https://docs.microsoft.com/en-us/dotnet/api/microsoft.office.interop.excel?redirectedfrom=MSDN&view=excel-pia) 需要电脑安装 [Microsoft Office](https://office.microsoft.com/) 并引用COM组件才可以使用（已知调用打印机需引用COM组件），所以选用类库可独立于Office组件，在任意一台电脑也可以运行。
[NPOI](https://github.com/tonyqus/npoi)：POI Java项目的.NET版本。可以非常轻松地读/写Office 2003/2007文件。
[DocX](https://github.com/xceedsoftware/docx)：DocX是一个.NET库，允许开发人员以简单直观的方式操作Word文件。

#### Excel文件操作
[ExcelHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/NPOI.Helper/Excel/ExcelHelper.cs)提供创建文件（2003/2007）及Sheet分页创建编辑，读取Excel文件至内存DataSet及反向DataSet保存至Excel文件。仅显示最外层引用方法，详细调用请在帮助类种查看！
``` CSharp
/// <summary>
/// Excel所有分页转换为DataSet
/// </summary>
/// <param name="strDataSourcePath">Excel文件路径</param>
/// <returns>成功返回Excel的DataSet,失败返回NULL</returns>
public static DataSet ExcelConversionDataSet(string strDataSourcePath)
{
    try
    {
        if (string.IsNullOrEmpty(strDataSourcePath) || !File.Exists(strDataSourcePath))
        {
            return null;
        }
        DataSet dsTargetData = new DataSet();
        Dictionary<int, string> dicAllSheet = GetExcelAllSheet(strDataSourcePath);
        foreach (var vAllSheet in dicAllSheet)
        {
            DataTable dtTargetData = new DataTable();
            dtTargetData.TableName = vAllSheet.Value;
            dtTargetData = ExcelConversionDataTable(strDataSourcePath, vAllSheet.Value);
            if (dtTargetData == null)
            {
                continue;
            }
            dsTargetData.Tables.Add(dtTargetData);
        }
        return dsTargetData;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return null;
    }
}
```

``` CSharp
/// <summary>
/// DataSet转换为Excel
/// 存在文件则新建DataTableName的分页(如果分页名冲突则或为空则使用默认名称)
/// 不存在文件则新建(Excel,名称为DataTableName,如果没有则使用默认名称)
/// </summary>
/// <param name="strDataSourcePath">Excel文件路径</param>
/// <param name="dsSourceData">DataTable数据</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool DataSetConversionExcel(string strDataSourcePath, DataSet dsSourceData)
{
    try
    {
        if (string.IsNullOrEmpty(strDataSourcePath) || dsSourceData.Tables.Count < 1)
        {
            return false;
        }
        foreach (DataTable dtSourceData in dsSourceData.Tables)
        {
            Dictionary<int, string> dicAllSheet = GetExcelAllSheet(strDataSourcePath);
            string strTableName = string.IsNullOrEmpty(dtSourceData.TableName) ? string.Format("Sheet{0}", dicAllSheet.Count + 1) : dtSourceData.TableName;
            if (dicAllSheet.ContainsValue(dtSourceData.TableName))
            {
                RemoveExcelSheet(strDataSourcePath, dtSourceData.TableName);
            }
            if (!FillDataTable(strDataSourcePath, strTableName, dtSourceData, true, 0, 0))
            {
                return false;
            }
        }
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}
```

根据公司项目需要，把多个Excel的Sheet页的内容及样式合并为一个文件，Microsoft.Office.Interop.Excel提供拷贝分页方法，但是需要安装Microsoft Office，所以用NPOI类库实现了一个拷贝方法。
``` CSharp
/// <summary>
/// 拷贝Sheet页到另一个Sheet页
/// </summary>
/// <param name="strSourceExcelPath">源Excel路径</param>
/// <param name="strFromSheetName">源Excel拷贝Sheet</param>
/// <param name="strTargetExcelPath">目标Excel路径</param>
/// <param name="strToSheetName">目标Excel拷贝Sheet</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool CopySheet(string strSourceExcelPath, string strFromSheetName, string strTargetExcelPath, string strToSheetName)
{
    try
    {
        if (string.IsNullOrEmpty(strSourceExcelPath) || string.IsNullOrEmpty(strTargetExcelPath) || !File.Exists(strSourceExcelPath))
        {
            TXTHelper.Logs(string.Format("源数据和目标数据参数为空或文件不存在!"));
            return false;
        }
        if (string.IsNullOrEmpty(strFromSheetName) || string.IsNullOrEmpty(strToSheetName))
        {
            TXTHelper.Logs(string.Format("源Sheet页和目标Sheet页参数为空!"));
            return false;
        }
        //获得源数据和目标数据的Sheet页
        IWorkbook iSourceWorkbook = null;
        ISheet iSourceSheet = GetExcelSheetAt(strSourceExcelPath, strFromSheetName, out iSourceWorkbook);
        IWorkbook iTargetWorkbook = null;
        ISheet iTargetSheet = null;
        if (iSourceSheet == null)
        {
            TXTHelper.Logs(string.Format("指定源数据Sheet页为空!"));
            return false;
        }
        if (!File.Exists(strTargetExcelPath))
        {
            //如果文件不存在则创建Excel
            if (System.IO.Path.GetExtension(strTargetExcelPath) == ".xls")
            {
                bool bCreare = CreateExcel_Office2003(strTargetExcelPath, strToSheetName);
            }
            else if (System.IO.Path.GetExtension(strTargetExcelPath) == ".xlsx")
            {
                bool bCreare = CreateExcel_Office2007(strTargetExcelPath, strToSheetName);
            }
            else
            {
                TXTHelper.Logs(string.Format("指定目标Excel文件路径格式错误!"));
                return false;
            }
            iTargetSheet = GetExcelSheetAt(strTargetExcelPath, strToSheetName, out iTargetWorkbook);
        }
        else
        {
            //如果文件存在则判断是否存在执行Sheet
            Dictionary<int, string> dicAllSheet = GetExcelAllSheet(strTargetExcelPath);
            if (dicAllSheet.ContainsValue(strToSheetName))
            {
                iTargetSheet = GetExcelSheetAt(strTargetExcelPath, strToSheetName, out iTargetWorkbook);
            }
            else
            {
                iTargetSheet = CreateExcelSheetAt(strTargetExcelPath, strToSheetName, out iTargetWorkbook);
            }
        }
        //调用Sheet拷贝Sheet方法
        bool bCopySheet = CopySheetAt(iSourceWorkbook, iSourceSheet, iTargetWorkbook, iTargetSheet);
        if (bCopySheet)
        {
            if (System.IO.Path.GetExtension(strTargetExcelPath) == ".xls")
            {
                FileStream fileStream2003 = new FileStream(Path.ChangeExtension(strTargetExcelPath, "xls"), FileMode.Create);
                iTargetWorkbook.Write(fileStream2003);
                fileStream2003.Close();
                iTargetWorkbook.Close();
            }
            else if (System.IO.Path.GetExtension(strTargetExcelPath) == ".xlsx")
            {
                FileStream fileStream2007 = new FileStream(Path.ChangeExtension(strTargetExcelPath, "xlsx"), FileMode.Create);
                iTargetWorkbook.Write(fileStream2007);
                fileStream2007.Close();
                iTargetWorkbook.Close();
            }
            return true;
        }
        else
        {
            TXTHelper.Logs(string.Format("拷贝失败!"));
            return false;
        }
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}
```

``` CSharp
// <summary>
/// 拷贝Sheet页到另一个Sheet页(浅拷贝,不提供保存方法)
/// Office2003单Sheet页仅支持4000个样式
/// </summary>
/// <param name="iSourceWorkbook">源Excel工作簿</param>
/// <param name="iFromSheet">源Sheet页</param>
/// <param name="iTargetWorkbook">目标Excel工作簿</param>
/// <param name="iToSheet">目标Sheet页</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool CopySheetAt(IWorkbook iSourceWorkbook, ISheet iFromSheet, IWorkbook iTargetWorkbook, ISheet iToSheet)
{
    try
    {
        //拷贝数据
        DataTable dtExcelFromData = GetDataTable(iFromSheet, false, 0, 0, 0, 0);
        iToSheet = FillDataTable(iToSheet, dtExcelFromData, false, 0, 0);
        //拷贝单元格合并
        for (int iMergedRegions = 0; iMergedRegions < iFromSheet.NumMergedRegions; iMergedRegions++)
        {
            iToSheet.AddMergedRegion(iFromSheet.GetMergedRegion(iMergedRegions));
        }
        //拷贝样式(遍历Sheet页行)
        List<ICellStyle> listCellStyle = new List<ICellStyle>();
        for (int iRowNum = 0; iRowNum <= iFromSheet.LastRowNum; iRowNum++)
        {
            IRow iFromRowData = iFromSheet.GetRow(iRowNum);
            IRow iToRowData = iToSheet.GetRow(iRowNum);
            if (iFromRowData == null || iToRowData == null)
            {
                continue;
            }
            //设置行高
            short sFromHeight = iFromRowData.Height;
            iToRowData.Height = sFromHeight;
            //遍历Sheet页列
            for (int iRowCell = 0; iRowCell <= iFromRowData.LastCellNum; iRowCell++)
            {
                //设置列宽
                int iFromColumnWidth = iFromSheet.GetColumnWidth(iRowNum) / 256;
                iToSheet.SetColumnWidth(iRowNum, iFromColumnWidth * 256);
                //复制数据
                ICell iFromCell = iFromRowData.GetCell(iRowCell);
                if (iFromCell != null)
                {
                    //获得源Sheet页的样式
                    ICellStyle iFromCellStyle = iFromCell.CellStyle;
                    //获得目标Excel指定Cell
                    ICell iToCell = iToRowData.GetCell(iRowCell);
                    if (iToCell == null) continue;
                    #region 复制单元格样式
                    //指定Cell创新目标Excel工作簿新样式
                    ICellStyle iToNewCellStyle = null;
                    foreach (ICellStyle vCellStyle in listCellStyle)
                    {
                        IFont iVToFont = vCellStyle.GetFont(iTargetWorkbook);
                        IFont iFromFont = iFromCellStyle.GetFont(iSourceWorkbook);
                        if (vCellStyle.Alignment == iFromCellStyle.Alignment &&
                            vCellStyle.BorderBottom == iFromCellStyle.BorderBottom &&
                            vCellStyle.BorderLeft == iFromCellStyle.BorderLeft &&
                            vCellStyle.BorderRight == iFromCellStyle.BorderRight &&
                            vCellStyle.BorderTop == iFromCellStyle.BorderTop &&
                            vCellStyle.BottomBorderColor == iFromCellStyle.BottomBorderColor &&
                            vCellStyle.DataFormat == iFromCellStyle.DataFormat &&
                            vCellStyle.FillBackgroundColor == iFromCellStyle.FillBackgroundColor &&
                            vCellStyle.FillForegroundColor == iFromCellStyle.FillForegroundColor &&
                            vCellStyle.FillPattern == iFromCellStyle.FillPattern &&
                            vCellStyle.Indention == iFromCellStyle.Indention &&
                            vCellStyle.IsHidden == iFromCellStyle.IsHidden &&
                            vCellStyle.IsLocked == iFromCellStyle.IsLocked &&
                            vCellStyle.LeftBorderColor == iFromCellStyle.LeftBorderColor &&
                            vCellStyle.RightBorderColor == iFromCellStyle.RightBorderColor &&
                            vCellStyle.Rotation == iFromCellStyle.Rotation &&
                            vCellStyle.TopBorderColor == iFromCellStyle.TopBorderColor &&
                            vCellStyle.VerticalAlignment == iFromCellStyle.VerticalAlignment &&
                            vCellStyle.WrapText == iFromCellStyle.WrapText &&
                            //字体比对
                            iVToFont.Color == iFromFont.Color &&
                            iVToFont.FontHeightInPoints == iFromFont.FontHeightInPoints &&
                            iVToFont.FontName == iFromFont.FontName &&
                            iVToFont.IsBold == iFromFont.IsBold &&
                            iVToFont.IsItalic == iFromFont.IsItalic &&
                            iVToFont.IsStrikeout == iFromFont.IsStrikeout &&
                            iVToFont.Underline == iFromFont.Underline)
                        {
                            iToNewCellStyle = vCellStyle;
                            break;
                        }
                    }
                    if (iToNewCellStyle == null)
                    {
                        //创建新样式
                        iToNewCellStyle = iTargetWorkbook.CreateCellStyle();
                        //复制样式
                        iToNewCellStyle.Alignment = iFromCellStyle.Alignment;//对齐
                        iToNewCellStyle.BorderBottom = iFromCellStyle.BorderBottom;//下边框
                        iToNewCellStyle.BorderLeft = iFromCellStyle.BorderLeft;//左边框
                        iToNewCellStyle.BorderRight = iFromCellStyle.BorderRight;//右边框
                        iToNewCellStyle.BorderTop = iFromCellStyle.BorderTop;//上边框
                        iToNewCellStyle.BottomBorderColor = iFromCellStyle.BottomBorderColor;//下边框颜色
                        iToNewCellStyle.DataFormat = iFromCellStyle.DataFormat;//数据格式
                        iToNewCellStyle.FillBackgroundColor = iFromCellStyle.FillBackgroundColor;//填充背景色
                        iToNewCellStyle.FillForegroundColor = iFromCellStyle.FillForegroundColor;//填充前景色
                        iToNewCellStyle.FillPattern = iFromCellStyle.FillPattern;//填充图案
                        iToNewCellStyle.Indention = iFromCellStyle.Indention;//压痕
                        iToNewCellStyle.IsHidden = iFromCellStyle.IsHidden;//隐藏
                        iToNewCellStyle.IsLocked = iFromCellStyle.IsLocked;//锁定
                        iToNewCellStyle.LeftBorderColor = iFromCellStyle.LeftBorderColor;//左边框颜色
                        iToNewCellStyle.RightBorderColor = iFromCellStyle.RightBorderColor;//右边框颜色
                        iToNewCellStyle.Rotation = iFromCellStyle.Rotation;//旋转
                        iToNewCellStyle.TopBorderColor = iFromCellStyle.TopBorderColor;//上边框颜色
                        iToNewCellStyle.VerticalAlignment = iFromCellStyle.VerticalAlignment;//垂直对齐
                        iToNewCellStyle.WrapText = iFromCellStyle.WrapText;//文字换行
                        //复制字体
                        IFont iFromFont = iFromCellStyle.GetFont(iSourceWorkbook);
                        IFont iToFont = iTargetWorkbook.CreateFont();
                        iToFont.Color = iFromFont.Color;//颜色
                        iToFont.FontHeightInPoints = iFromFont.FontHeightInPoints;//字号
                        iToFont.FontName = iFromFont.FontName;//字体
                        iToFont.IsBold = iFromFont.IsBold;//加粗
                        iToFont.IsItalic = iFromFont.IsItalic;//斜体
                        iToFont.IsStrikeout = iFromFont.IsStrikeout;//删除线
                        iToFont.Underline = iFromFont.Underline;//下划线
                        iToNewCellStyle.SetFont(iToFont);
                        //保存到缓存集合中
                        listCellStyle.Add(iToNewCellStyle);
                    }
                    //复制样式到指定表格中
                    iToCell.CellStyle = iToNewCellStyle;
                    #endregion
                }
            }
        }
        return true;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return false;
    }
}
```

#### Word文件操作
[WordHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/NPOI.Helper/Word/WordHelper.cs)提供创建文件（2003/2007）及替换段落表格标签（匹配替换'{标签}','#标签#'），替换图片功能。仅显示最外层引用方法，详细调用请在帮助类种查看！
``` CSharp
/// <summary>
/// 替换文本标签
/// </summary>
/// <param name="strDataSourcePath">Word文件路径</param>
/// <param name="strLabelName">标签名称(带标签符号)</param>
/// <param name="strReplaceLabel">替换标签文本</param>
/// <returns>成功返回替换数量,失败返回-1</returns>
public static int ReplaceTextLabel(string strDataSourcePath, string strLabelName, string strReplaceLabel)
{
    try
    {
        if (string.IsNullOrEmpty(strDataSourcePath) || !File.Exists(strDataSourcePath) || string.IsNullOrEmpty(strLabelName) || string.IsNullOrEmpty(strReplaceLabel))
        {
            return -1;
        }
        int iNumber = 0;
        FileStream fileStreamOpen = new FileStream(strDataSourcePath, FileMode.Open, FileAccess.Read);
        XWPFDocument wordDocument = new XWPFDocument(fileStreamOpen);
        foreach (XWPFParagraph wordParagraph in wordDocument.Paragraphs)
        {
            if (wordParagraph.ParagraphText.IndexOf(strLabelName) >= 0)
            {
                string strReplaceTextLabel = wordParagraph.ParagraphText.Replace(strLabelName, strReplaceLabel);
                foreach (XWPFRun wordRun in wordParagraph.Runs)
                {
                    wordRun.SetText(string.Empty, 0);
                }
                wordParagraph.CreateRun().SetText(strReplaceTextLabel, 0);
                iNumber++;
            }
        }
        foreach (XWPFTable wordTable in wordDocument.Tables)
        {
            foreach (XWPFTableRow wordTableRow in wordTable.Rows)
            {
                foreach (XWPFTableCell wordTableCell in wordTableRow.GetTableCells())
                {
                    foreach (XWPFParagraph wordParagraph in wordTableCell.Paragraphs)
                    {
                        if (wordParagraph.ParagraphText.IndexOf(strLabelName) >= 0)
                        {
                            string strReplaceTextLabel = wordParagraph.ParagraphText.Replace(strLabelName, strReplaceLabel);
                            foreach (XWPFRun wordRun in wordParagraph.Runs)
                            {
                                wordRun.SetText(string.Empty, 0);
                            }
                            wordParagraph.CreateRun().SetText(strReplaceTextLabel, 0);
                            iNumber++;
                        }
                    }
                }
            }
        }
        FileStream fileStreamSave = new FileStream(strDataSourcePath, FileMode.Create);
        wordDocument.Write(fileStreamSave);
        fileStreamSave.Close();
        wordDocument.Close();
        return iNumber;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return -1;
    }
}
```

``` CSharp
/// <summary>
/// 替换表格标签(DataTable替换)
/// </summary>
/// <param name="strDataSourcePath">Word文件路径</param>
/// <param name="strLabelName">标签名称(带标签符号)</param>
/// <param name="dtReplaceLabel">替换标签DataTable</param>
/// <returns>成功返回1,失败返回-1</returns>
public static int ReplaceDataTableLabel(string strDataSourcePath, string strLabelName, DataTable dtReplaceLabel)
{
    try
    {
        if (string.IsNullOrEmpty(strDataSourcePath) || !File.Exists(strDataSourcePath) || string.IsNullOrEmpty(strLabelName) || dtReplaceLabel == null || dtReplaceLabel.Rows.Count < 1)
        {
            return -1;
        }
        FileStream fileStreamOpen = new FileStream(strDataSourcePath, FileMode.Open, FileAccess.Read);
        XWPFDocument wordDocument = new XWPFDocument(fileStreamOpen);
        int iLableRowPosition = -1;
        int iLableCellPosition = -1;
        foreach (XWPFTable wordTable in wordDocument.Tables)
        {
            for (int iTableRow = 0; iTableRow < wordTable.Rows.Count; iTableRow++)
            {
                for (int iTableCell = 0; iTableCell < wordTable.Rows[iTableRow].GetTableCells().Count; iTableCell++)
                {
                    foreach (XWPFParagraph wordParagraph in wordTable.Rows[iTableRow].GetTableCells()[iTableCell].Paragraphs)
                    {
                        if (wordParagraph.ParagraphText.IndexOf(strLabelName) >= 0)
                        {
                            if (iLableRowPosition < 0 && iLableCellPosition < 0)
                            {
                                iLableRowPosition = iTableRow;
                                iLableCellPosition = iTableCell;
                            }
                        }
                        if (iLableRowPosition >= 0 && iLableCellPosition >= 0)
                        {
                            int iCurrentRow = iTableRow - iLableRowPosition;
                            int iCurrentCell = iTableCell - iLableCellPosition;
                            if ((iCurrentRow < dtReplaceLabel.Rows.Count && iCurrentRow >= 0) && (iCurrentCell < dtReplaceLabel.Columns.Count && iCurrentCell >= 0))
                            {
                                foreach (XWPFRun wordRun in wordParagraph.Runs)
                                {
                                    wordRun.SetText(string.Empty, 0);
                                }
                                wordParagraph.CreateRun().SetText(dtReplaceLabel.Rows[iCurrentRow][iCurrentCell].ToString(), 0);
                            }
                        }
                    }
                }
            }
        }
        FileStream fileStreamSave = new FileStream(strDataSourcePath, FileMode.Create);
        wordDocument.Write(fileStreamSave);
        fileStreamSave.Close();
        wordDocument.Close();
        return 1;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return -1;
    }
}
```

``` CSharp
/// <summary>
/// 替换图片标签(使用DocX.dll类库,调用这个方法后NPOI无法读取文档)
/// </summary>
/// <param name="strDataSourcePath">Word文件路径</param>
/// <param name="strLabelName">标签名称(带标签符号)</param>
/// <param name="strImagePath">替换的图片路径</param>
/// <param name="iImageWidth">替换的图片宽度(小于0则显示原图宽度)</param>
/// <param name="iImageHeight">替换的图片高度(小于0则显示原图高度)</param>
/// <returns>成功返回替换数量,失败返回-1</returns>
public static int ReplaceImageLabel(string strDataSourcePath, string strLabelName, string strImagePath, int iImageWidth, int iImageHeight)
{
    try
    {
        if (string.IsNullOrEmpty(strDataSourcePath) || !File.Exists(strDataSourcePath) || string.IsNullOrEmpty(strLabelName) || string.IsNullOrEmpty(strImagePath) || !File.Exists(strImagePath))
        {
            return -1;
        }
        int iNumber = 0;
        //使用DocX.dll类库
        DocX mDocX = DocX.Load(strDataSourcePath);
        //遍历段落
        foreach (Paragraph wordParagraph in mDocX.Paragraphs)
        {
            if (wordParagraph.Text.IndexOf(strLabelName) >= 0)
            {
                //添加图片
                Novacode.Image pImag = mDocX.AddImage(strImagePath);
                Picture pPicture = pImag.CreatePicture();
                //如果传入宽度小于0,则以原始大小插入
                if (iImageWidth >= 0)
                {
                    pPicture.Width = iImageWidth;
                }
                //如果传入高度小于0,则以原始大小插入
                if (iImageHeight >= 0)
                {
                    pPicture.Height = iImageHeight;
                }
                //将图像插入到段落后面
                wordParagraph.InsertPicture(pPicture);
                //清空文本(清空放在前面会导致替换失败文字消失)
                wordParagraph.ReplaceText(strLabelName, string.Empty);
                iNumber++;
            }
        }
        mDocX.SaveAs(strDataSourcePath);
        return iNumber;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
        return -1;
    }
}
```