---
title: SwiftUI 隐藏 NavigationView 导航栏
date: 2020-11-30 13:05:00
tags: [swift,swiftui]
categories: Swift
---
### SwiftUI 对 NavigationView 导航栏的一些操作
<!-- more -->
#### 简介
原本简单的一个需求，但是相关的文档极少，所以还是记录一下的好
#### 需求描述
在 SwiftUI 开发 IOS 应用时，官方建议使用 [NavigationView](https://developer.apple.com/documentation/swiftui/navigationview) 来跳转页面，使用环境例如登录页弹出注册与修改密码二级菜单，登陆页需要隐藏导航栏，注册与修改密码菜单则需要显示导航栏显示标题并且可以返回登录页。
#### 修改前截图示例

#### 修改前代码片段
``` Swift
struct LoginView: View {
    var body: some View {
        NavigationView{
            VStack(){

                ……

                HStack(){
                    NavigationLink(
                        destination: RegisterView(),
                        label: {
                            Text("注册用户")
                        })
                    Spacer()
                    NavigationLink(
                        destination: RetrievePasswordView(),
                        label: {
                            Text("找回密码")
                        })
                }
                .padding(EdgeInsets(top: 30, leading: 40, bottom: 0, trailing: 40))
            }
            .navigationBarTitle(Text("登录"))
        }
    }
}
```

#### 修改后截图示例

#### 修改后代码片段
``` Swift
struct LoginView: View {

    @State var isNavigationBarHidden: Bool = true
  
    var body: some View {
        NavigationView{
            VStack(){

                ……

                HStack(){
                    NavigationLink(
                        destination: RegisterView(),
                        label: {
                            Text("注册用户")
                        })
                    Spacer()
                    NavigationLink(
                        destination: RetrievePasswordView(),
                        label: {
                            Text("找回密码")
                        })
                }
                .padding(EdgeInsets(top: 30, leading: 40, bottom: 0, trailing: 40))
            }
            .navigationBarTitle(Text("登录"))
            .navigationBarHidden(self.isNavigationBarHidden)
            .onAppear {
                self.isNavigationBarHidden = true
            }
            .onDisappear {
                self.isNavigationBarHidden = false
            }
        }
    }
}
```