---
title: Excel 单元格解密
date: 2021-08-15 14:30:18
tags: [c#,excel]
categories: C#.Net
---
### 解决 Excel 单元格显示与复制不一样的情况
<!-- more -->
#### 简介
<img src="https://sadness96.github.io/images/blog/csharp-ExcelCellDecryption/ExcelCellDecryption.png"/>

收到一份 Excel 文件，表面看起来一切正常，但是表格内所有数据单元格显示与表格上方编辑栏内容不符，复制单元格到记事本，显示内容与编辑栏一致，看起来是有人刻意对单元格文本进行加盐处理，应该是为了防止拷贝直接使用数据。

#### 参考
帖子 [excel单元格怎么让实际值与显示的值一致](https://www.52pojie.cn/thread-1456979-1-1.html) 与 [电子表格里的数据如何去除隐藏部分](https://www.52pojie.cn/thread-1433060-1-1.html) 中有遇到相同情况，当中有人给出了原理，但是并没有提供一种快速的解决办法。

#### 验证
<img src="https://sadness96.github.io/images/blog/csharp-ExcelCellDecryption/NotepadCellText.png"/>

拷贝一个单元格到记事本中，可以看到显示内容与 Excel 编辑中一致为加盐后的结果。

<img src="https://sadness96.github.io/images/blog/csharp-ExcelCellDecryption/WordCellText.png"/>

拷贝一个单元格数据到 Word 中，可以看到显示内容正确，但是仔细看左下角开头部位有几个字号为1磅或2磅不易察觉的宽度。

#### 解决办法
##### 处理前准备
1. 前文参考的帖子中有提到替换的方式，我尝试在 Excel 中按照字号替换，最终失败，一格一格数据拷贝到 Word 中替换，这个数据量着实劝退。
1. 使用 NPOI 读取单元格格式，替换其中的文本，但是在长时候发现 NPOI 对单元格中多种字体大小颜色很难判断，最终放弃。
1. Excel 实际为压缩文件固定格式，可以解压缩获取其中 XML 文件，从而修改，解压缩后看到单元格数据都储存在 "\xl\sharedStrings.xml" 文件中，但是不知为何，这个 XML 文件的节点并不完整，解析时会报错，修复文件是个不小的工作量，并且格式恢复为 .xlsx 后 Excel 报文件损坏，所以放弃。
1. 通过另存的方式保存为 XML 格式后再做解析，另存为 "XML 数据"，提示错误:"工作簿不包含任何 XML 映射"，所以另存为 "XML 电子表格 2003"，可以看到每个单元格内的文字格式。

``` xml
<Cell ss:StyleID="s68">
    <ss:Data ss:Type="String" xmlns="http://www.w3.org/TR/REC-html40">
        <Font html:Color="#FFFFF2">a1</Font>
        <Font html:Size="9">13763569</Font>
        <Font html:Size="1" html:Color="#FFFFCC">8</Font>
        <Font html:Size="9">999</Font>
    </ss:Data>
</Cell>
```

##### 判断加盐内容
多找几条数据后发现加盐的干扰项为（字体字号为 1磅 或 2磅，颜色为不易察觉的白色）：
``` txt
html:Color="#FFFFF2"
html:Color="#FFFFF1"
html:Color="#FFFFCC"
html:Color="#FFFFFF"
html:Size="1"
html:Size="2"
```
##### 处理文件代码
完整代码：[ExcelCellDecryption](https://github.com/Sadness96/ExcelCellDecryption)
程序运行选择另存为 "XML 电子表格 2003" 的 XML 文件，点击执行。

``` csharp
// 解析 Excel XML 文档
XmlDocument doc = new XmlDocument();
doc.Load(vNewFile);
XmlNamespaceManager nsmgr = new XmlNamespaceManager(doc.NameTable);
nsmgr.AddNamespace("ab", "http://www.w3.org/TR/REC-html40");
nsmgr.AddNamespace("ss", "urn:schemas-microsoft-com:office:spreadsheet");
// 删除掺杂的数据
XmlNodeList nodeFonts = doc.SelectNodes("//ab:Font", nsmgr);
for (int i = 0; i < nodeFonts.Count; i++)
{
    var vXmlNodeFont = nodeFonts[i];
    bool bIsRemove = false;
    foreach (var item in listRemoveIdentification)
    {
        if (vXmlNodeFont.OuterXml.Contains(item))
        {
            bIsRemove = true;
            break;
        }
    }
    if (bIsRemove)
    {
        var vParentNode = vXmlNodeFont.ParentNode;
        vParentNode.RemoveChild(vXmlNodeFont);
    }
}
// 合并整理后的数据
XmlNodeList nodeDatas = doc.SelectNodes("//ss:Data", nsmgr);
for (int i = 0; i < nodeDatas.Count; i++)
{
    var vXmlNodeData = nodeDatas[i];
    var vXmlNodeFonts = vXmlNodeData.ChildNodes;
    if (vXmlNodeFonts.Count >= 2)
    {
        // Data 中 Font 数量大于等于 2 需要合并
        string strTxt = "";
        XmlNode xmlNodeMain = null;
        List<XmlNode> xmlNodesPrepare = new List<XmlNode>();
        // 记录数据 拼接文本 记录主要 Font 和需要删除的 Font
        for (int j = 0; j < vXmlNodeFonts.Count; j++)
        {
            var vXmlNodeFont = vXmlNodeFonts[j];
            if (j == 0)
            {
                xmlNodeMain = vXmlNodeFont;
            }
            else
            {
                xmlNodesPrepare.Add(vXmlNodeFont);
            }
            strTxt += vXmlNodeFont.InnerText;
        }
        // 记录主要 Font,超过15位增加 "'"
        if (strTxt.Length >= 15 && IsNumeric(strTxt) && !strTxt.First().Equals('\''))
        {
            xmlNodeMain.InnerText = $"'{strTxt}";
        }
        else
        {
            xmlNodeMain.InnerText = strTxt;
        }
        // 删除的 Font
        var vParentNode = xmlNodeMain.ParentNode;
        for (int k = 0; k < xmlNodesPrepare.Count; k++)
        {
            vParentNode.RemoveChild(xmlNodesPrepare[k]);
        }
    }
}
doc.Save(vNewFile);
```

##### 手动处理
1. 执行完成后会生成："{XXX}_decrypt.xml" 文件，使用 Excel 打开。
1. 修改整表字号为标准大小（我这里是宋体9号）。
1. Ctrl+H 打开查找和替换，替换所有 " " 为 ""。
1. 另存文件为 .xlsx 格式，处理结束。

#### 如何制作这样的数据
既然解决了问题，那么在按照原路制造出来也是比较简单的，比如直接修改 xml 文件在 Cell 单元格中添加不易察觉的 Font，或者参考帖子 [NPOI Excel同一个单元格 多种字体](https://www.cnblogs.com/leoxjy/p/10669924.html) 使用 NPOI 写入即可。