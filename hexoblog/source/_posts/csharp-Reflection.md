---
title: C# Reflection(反射)
date: 2016-12-10 23:10:21
tags: [c#,reflection]
categories: C#.Net
---
### 通过反射动态调用方法

<!-- more -->
### 简介
[System.Reflection](https://docs.microsoft.com/zh-cn/dotnet/api/system.reflection?view=netframework-4.5) 命名空间包含通过检查托管代码中程序集、模块、成员、参数和其他实体的元数据来检索其相关信息的类型。 这些类型还可用于操作加载类型的实例，例如挂钩事件或调用方法。 
↑上边是微软 [MSDN](https://docs.microsoft.com/zh-cn/) 的介绍，balabala一大堆…，总之反射调用它就对了，通过类库类名以及方法名称调用方法。

### 使用
#### 通过反射运行方法
##### 调用方法代码
``` csharp
/// <summary>
/// 运行方法
/// </summary>
/// <param name="strLibraryName">库名</param>
/// <param name="strClassName">类名(带路径)</param>
/// <param name="strMethodName">方法名</param>
/// <param name="parameters">参数</param>
/// <returns>返回结果</returns>
public static object RunMethod(string strLibraryName, string strClassName, string strMethodName, object[] parameters)
{
    try
    {
        Type type = Type.GetType($"{strLibraryName}.{strClassName}");
        if (type != null)
        {
            var vMethod = type.GetMethod(strMethodName);
            if (vMethod != null)
            {
                return vMethod.Invoke(Activator.CreateInstance(type), parameters);
            }
        }
        return null;
    }
    catch (Exception)
    {
        return null;
    }
}
```

##### 调用方法示例
``` csharp
var parameters = new object[] { item };
RunMethod("ReflectionProject_Test", "ClassName", "MethodName", parameters);
```

#### 通过反射接口插件式开发
##### 定义接口
``` csharp
/// <summary>
/// 菜单插件接口
/// </summary>
public interface MenuPluginInterface
{
    /// <summary>
    /// 菜单单击事件
    /// </summary>
    void Click();

    /// <summary>
    /// 功能名称
    /// </summary>
    string strFunctionName { get; }

    /// <summary>
    /// 功能分组
    /// </summary>
    string strFunctionGroup { get; }
}
```

##### 插件继承接口
``` csharp
/// <summary>
/// 功能按钮测试Command
/// </summary>
public class TestCommand : MenuPluginInterface
{
    /// <summary>
    /// Click Command
    /// </summary>
    public void Click()
    {
        // 打开插件功能
    }

    /// <summary>
    /// 功能名称
    /// </summary>
    public string strFunctionName
    {
        get { return "测试功能"; }
    }

    /// <summary>
    /// 功能分组
    /// </summary>
    public string strFunctionGroup
    {
        get { return "测试分组"; }
    }
}
```

##### 反射方法
``` csharp
/// <summary>
/// 运行插件窗体
/// </summary>
/// <param name="strDllPath">Dll路径</param>
/// <param name="strClassName">全类名</param>
/// <returns>成功返回true,失败返回false</returns>
public static bool RunPluginClick(string strDllPath, string strClassName)
{
    try
    {
        //反射获得Class Type
        Assembly assembly = Assembly.LoadFrom(strDllPath);
        Type type = assembly.GetType(strClassName);
        if (type != null)
        {
            var container = new UnityContainer();
            container.RegisterType<MenuPluginInterface>(new ContainerControlledLifetimeManager());
            container.RegisterType(typeof(MenuPluginInterface), type);
            var manager = container.Resolve<MenuPluginInterface>();
            manager.Click();
            return true;
        }
        else
        {
            return false;
        }
    }
    catch (Exception)
    {
        return false;
    }
}
```

##### 通过反射打开插件
``` csharp
string strProjectDll = "ReflectionProject_Test.dll";
RunPluginClick(strProjectDll, "ReflectionProject_Test.TestCommand");
```