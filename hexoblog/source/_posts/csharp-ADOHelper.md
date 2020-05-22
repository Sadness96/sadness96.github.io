---
title: ADO.NET 帮助类
date: 2016-12-21 13:30:20
tags: [c#,helper,ado.net,sql server,oracle,mysql,access,sqlite]
categories: C#.Net
---
<img src="https://sadness96.github.io/images/blog/csharp-DevFramework/%E6%95%B0%E6%8D%AE%E5%BA%93%E8%BD%AC%E6%8D%A2%E5%B7%A5%E5%85%B7.png"/>

### 使用 ADO.NET 二次封装ORM框架的数据库操作帮助类
<!-- more -->
#### 简介
工作中大量需要多种不同数据格式互相转换，通过ADO.NET实现可视化数据转换工具，目前支持关系型数据库SqlServer、Oracle、MySql、Access、SQLite。
#### 设计
简易的ORM框架，多种数据库操作封装为一套帮助类中，后期使用不需要过多考虑数据库类型，以及减少在代码中拼写SQL语句。近乎通用的连接方式以及增删改查，支持事务处理。
#### 帮助类、官方文档及其调用方式
##### SQLServer：
[SqlServerHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/ADO.Helper/SqlServer/SqlServerHelper.cs) 引用 [System.Data.SqlClient](https://technet.microsoft.com/zh-cn/system.data.sqlclient) 库
调用方式：
``` CSharp
SqlServerHelper sqlHelper = new SqlServerHelper();
sqlHelper.SqlServerConnectionString(string server, string database, string uid, string pwd);
sqlHelper.Open();
sqlHelper.Close();
```
##### Oracle：
[OracleHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/ADO.Helper/Oracle/OracleHelper.cs) 引用 [System.Data.OracleClient](https://technet.microsoft.com/zh-cn/system.data.oracleclient) 库
调用方式：
``` CSharp
OracleHelper sqlHelper = new OracleHelper();
sqlHelper.OracleConnectionString(string Source, string Id, string Password);
sqlHelper.Open();
sqlHelper.Close();
```
##### MySQL：
[MySqlHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/ADO.Helper/MySql/MySqlHelper.cs) 引用 [MySql.Data.MySqlClient](https://dev.mysql.com/doc/dev/connector-net/8.0/html/N_MySql_Data_MySqlClient.htm) 库
调用方式：
``` CSharp
MySqlHelper sqlHelper = new MySqlHelper();
sqlHelper.MySqlConnectionString(string server, string id, string password, string database);
sqlHelper.Open();
sqlHelper.Close();
```
##### Access：
[AccessHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/ADO.Helper/Access/AccessHelper.cs) 引用 [System.Data.OleDb](https://docs.microsoft.com/zh-cn/dotnet/api/system.data.oledb?redirectedfrom=MSDN&view=netframework-4.8) 库
调用方式：
``` CSharp
AccessHelper sqlHelper = new AccessHelper();
sqlHelper.AccessConnectionPath_Office2003(string source);
sqlHelper.AccessConnectionPath_Office2007(string source);
sqlHelper.Open();
sqlHelper.Close();
```
##### SQLite：
[SQLiteHelper](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/ADO.Helper/SQLite/SQLiteHelper.cs) 引用 [System.Data.SQLite](http://system.data.sqlite.org/) 库
调用方式：
``` CSharp
SQLiteHelper sqlHelper = new SQLiteHelper();
sqlHelper.SQLiteConnectionPath(string source);
sqlHelper.Open();
sqlHelper.Close();
```
#### 数据转换帮助类
由于每种数据库的字段类型、字符格式长度等不一致，所以专门写了一个用于互相兼容的帮助类，用于自动识别源数据库与目标数据库差异，自动修改。
[数据转换帮助类](https://github.com/Sadness96/Sadness/tree/master/Code/Helper/ADO.Helper/DatabaseConversion)
##### 删除DataTable中的空行
数据源以C#基础类型DataTable传递，在实际使用中存在空行导致异常
``` CSharp
/// <summary>
/// 删除DataTable中的空行
/// 弱引用,可直接修改参数
/// </summary>
/// <param name="dtDataSource">源数据(DataTable)</param>
/// <returns>删除空行后的DataTable</returns>
public static DataTable RemoveEmpty(DataTable dtDataSource)
{
    try
    {
        List<DataRow> listRemove = new List<DataRow>();
        for (int i = 0; i < dtDataSource.Rows.Count; i++)
        {
            bool IsNull = true;
            for (int j = 0; j < dtDataSource.Columns.Count; j++)
            {
                if (!string.IsNullOrEmpty(dtDataSource.Rows[i][j].ToString().Trim()))
                {
                    IsNull = false;
                }
            }
            if (IsNull)
            {
                listRemove.Add(dtDataSource.Rows[i]);
            }
        }
        for (int i = 0; i < listRemove.Count; i++)
        {
            dtDataSource.Rows.Remove(listRemove[i]);
        }
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
    }
    return dtDataSource;
}
```
##### DataTable与List<T>互相转换
实际使用中List<T>比DataTable更方便使用，提供互相转换方法方便开发
``` CSharp
/// <summary>
/// DataTable转换为List<T>
/// </summary>
/// <typeparam name="T">数据模型</typeparam>
/// <param name="dtDataSource">源数据(DataTable)</param>
/// <returns>成功返回List<T>,失败返回null</returns>
public static List<T> ConvertToList<T>(DataTable dtDataSource) where T : class,new()
{
    try
    {
        List<T> listT = new List<T>();
        foreach (DataRow drDataSource in dtDataSource.Rows)
        {
            T t = new T();
            PropertyInfo[] propertyInfos = t.GetType().GetProperties();
            foreach (PropertyInfo propertyInfo in propertyInfos)
            {
                string tempName = propertyInfo.Name;
                if (dtDataSource.Columns.Contains(tempName))
                {
                    if (!propertyInfo.CanWrite) continue;
                    object value = drDataSource[tempName];
                    if (value != DBNull.Value)
                    {
                        if (propertyInfo.GetMethod.ReturnParameter.ParameterType.Name == "Int32")
                        {
                            value = Convert.ToInt32(value);
                        }
                        propertyInfo.SetValue(t, value, null);
                    }
                }
            }
            listT.Add(t);
        }
        return listT;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
    }
    return null;
}
```
``` CSharp
/// <summary>
/// List<T>转换为DataTable
/// </summary>
/// <param name="listDataSource">源数据</param>
/// <returns>成功返回DataTable,失败返回null</returns>
public static DataTable ConvertDataTable(IList listDataSource)
{
    try
    {
        DataTable dataTable = new DataTable();
        if (listDataSource.Count > 0)
        {
            PropertyInfo[] propertyInfos = listDataSource[0].GetType().GetProperties();
            foreach (PropertyInfo propertyInfo in propertyInfos)
            {
                dataTable.Columns.Add(propertyInfo.Name, propertyInfo.PropertyType);
            }
            foreach (var vDataSource in listDataSource)
            {
                ArrayList arrayList = new ArrayList();
                foreach (PropertyInfo propertyInfo in propertyInfos)
                {
                    arrayList.Add(propertyInfo.GetValue(vDataSource, null));
                }
                dataTable.LoadDataRow(arrayList.ToArray(), true);
            }
        }
        return dataTable;
    }
    catch (Exception ex)
    {
        TXTHelper.Logs(ex.ToString());
    }
    return null;
}
```
##### String转Unicode,并去除'\\ufeff'非法字符
个别数据中存在有非法字符，避免异常，转换时批量处理
``` CSharp
/// <summary>
/// 去除非法字符'\\ufeff'
/// </summary>
/// <param name="strSource">数据源</param>
/// <returns>修正后的字符</returns>
public static string RemoveIllegal(string strSource)
{
    return UnicodeToString(StringToUnicode(strSource));
}

/// <summary>
/// String转Unicode,并去除'\\ufeff'非法字符
/// </summary>
/// <param name="strSource">数据源</param>
/// <returns>Unicode编码字符</returns>
public static string StringToUnicode(string strSource)
{
    StringBuilder stringBuilder = new StringBuilder();
    //先把字符串转换成 UTF-16 的Btye数组
    byte[] bytes = Encoding.Unicode.GetBytes(strSource);
    for (int i = 0; i < bytes.Length; i += 2)
    {
        //根据Unicode规则，每两个byte表示一个汉字，并且后前顺序，英文前面补00
        stringBuilder.AppendFormat("\\u{0}{1}", bytes[i + 1].ToString("x").PadLeft(2, '0'), bytes[i].ToString("x").PadLeft(2, '0'));
    }
    //去掉'?'的Unicode码,?=003f,Unicode以\u开头,\\为转义\
    return stringBuilder.Replace("\\ufeff", string.Empty).ToString();
}

/// <summary>
/// Unicode转String
/// </summary>
/// <param name="strSource">数据源</param>
/// <returns>String类型编码字符</returns>
public static string UnicodeToString(string strSource)
{
    return new Regex(@"\\u([0-9A-F]{4})", RegexOptions.IgnoreCase | RegexOptions.Compiled).Replace(strSource, x => string.Empty + Convert.ToChar(Convert.ToUInt16(x.Result("$1"), 16)));
}
```
##### 字段类型转换
由于每种数据库字段类型及字段长度和主键不一致，根据每种目标数据库做单独修改
代码过长，请查阅目录下方法 [TypeProcessing](https://github.com/Sadness96/Sadness/blob/master/Code/Helper/ADO.Helper/DatabaseConversion/TypeProcessing.cs)