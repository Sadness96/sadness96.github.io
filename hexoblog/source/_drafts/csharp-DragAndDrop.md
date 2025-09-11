---
title: Drag And Drop
date: 2025-09-11 15:38:15
tags: [c#,wpf,drop]
categories: C#.Net
---
### WPF 拖拽传递数据
<!-- more -->
### 简介
在桌面端软件中拖拽传递数据属于较为常见的功能，近期再次使用时做一下记录。
用户控件通过设置 AllowDrop="True" 开启是否允许拖拽释放，通过 DragDrop.DoDragDrop 函数传递数据。

### 代码
#### 拖拽源
示例为 TreeView 控件
``` XML
<TreeView x:Name="SourceTreeView"
          PreviewMouseLeftButtonDown="TreeView_PreviewMouseLeftButtonDown"
          MouseMove="TreeView_MouseMove">
    <TreeViewItem Header="Category 1">
        <TreeViewItem Header="Item 1.1"/>
        <TreeViewItem Header="Item 1.2"/>
    </TreeViewItem>
    <TreeViewItem Header="Category 2">
        <TreeViewItem Header="Item 2.1"/>
        <TreeViewItem Header="Item 2.2"/>
    </TreeViewItem>
</TreeView>
```

``` CSharp
/// <summary>
/// 拖拽 TreeViewItem
/// </summary>
private TreeViewItem? _draggedItem;

/// <summary>
/// 拖拽 TreeViewItem 起始坐标
/// </summary>
private Point _draggedStartPoint;

/// <summary>
/// 树控件隧道事件
/// 最先触发，用于捕获拖拽起始点和目标TreeViewItem。隧道事件从根元素向源元素传递，适合做全局拦截。
/// </summary>
/// <param name="sender"></param>
/// <param name="e"></param>
private void TreeView_PreviewMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
{
    if (e.OriginalSource is DependencyObject dependencyObject)
    {
        _draggedStartPoint = e.GetPosition(null);

        var clickedItem = e.OriginalSource as DependencyObject;
        while (clickedItem != null && !(clickedItem is TreeViewItem))
        {
            clickedItem = VisualTreeHelper.GetParent(clickedItem);
        }

        if (clickedItem is TreeViewItem item)
        {
            _draggedItem = item;
        }
    }
}

/// <summary>
/// 树控件冒泡事件
/// 检测鼠标移动距离超过阈值(5像素)后，调用DragDrop.DoDragDrop()启动系统拖拽操作。冒泡事件从源元素向根元素传递。
/// </summary>
/// <param name="sender"></param>
/// <param name="e"></param>
private void TreeView_MouseMove(object sender, MouseEventArgs e)
{
    if (e.LeftButton == MouseButtonState.Pressed && _draggedItem != null)
    {
        if ((e.GetPosition(null) - _draggedStartPoint).Length > 5)
        {
            DragDrop.DoDragDrop(_draggedItem, $"{_draggedItem.Header}", DragDropEffects.Copy);
        }
    }
}
```

#### 拖拽目标
示例为 Canvas 控件
``` XML
<Canvas x:Name="TargetCanvas" AllowDrop="True" Drop="Canvas_Drop" DragOver="Canvas_DragOver"/>
```

``` CSharp
/// <summary>
/// 拖放过程中的动态反馈
/// </summary>
/// <param name="sender"></param>
/// <param name="e"></param>
private void Canvas_DragOver(object sender, DragEventArgs e)
{
    e.Effects = DragDropEffects.Copy;
    e.Handled = true;
}

/// <summary>
/// 拖放操作的最终执行
/// </summary>
/// <param name="sender"></param>
/// <param name="e"></param>
private void Canvas_Drop(object sender, DragEventArgs e)
{
    var vDropPosition = e.GetPosition(TargetCanvas);

    if (e.Data.GetDataPresent(typeof(string)) && e.Data.GetData(typeof(string)) is string itemContent)
    {
        TextBlock newText = new TextBlock
        {
            Text = itemContent,
            Margin = new Thickness(vDropPosition.X, vDropPosition.Y, 0, 0)
        };
        TargetCanvas.Children.Add(newText);
    }
}
```