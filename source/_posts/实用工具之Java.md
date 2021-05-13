---
title: 实用工具之Java
date: 2019-10-13 15:21:11
author: Myhaa
img:
top: false
cover: false
coverImg:
password:
toc: true
mathjax: false
summary: 有关Java的笔记
categories: 实用工具
tags:
  - Java
  - 编程语言
---

# 一、Java基础

## 安装

### 名词解释

- `JDK`：Java Development Kit

  如果只有Java源码，要编译成Java字节码，就需要JDK，因为JDK除了包含JRE，还提供了编译器JavaC、调试器JDB等开发工具

- `JRE`：Java Runtime Environment

  JRE就是运行Java字节码的虚拟机

* `IDE`：Integrated Development Environment 

  流行的有：Eclipse、IntelliJ Idea

### 安装JDK

* [安装链接](<https://www.liaoxuefeng.com/wiki/1252599548343744/1280507291631649>)

### 安装IDE

* [IntelliJ Idea](<https://www.jetbrains.com/idea/>)

## 程序基础

### 基本结构

（1）一个完整的Java程序。

```java
/**
 * 可以用来自动创建文档的注释
 */
public class Hello {
    public static void main(String[] args) {
        // 向屏幕输出文本:
        final double PI = 3.14; // PI是一个常量
        System.out.println("Hello, world!");
        /* 多行注释开始
        注释内容
        注释结束 */
    }
} // class定义结束
```

（2）结构解释

- 一个程序的基本单位是`class`。类名要求必须以英文字母开头，后接字母，数字和下划线的组合，习惯以大写字母开头。
- `public`是访问修饰符，表示该class是公开的。
- `main`是方法名，返回值是`void`。
- 每一行语句后面都要加分号`;`。
- 一个文件顶多一个public类，且类名与文件名一致。
- `final`是常量修饰符。

### 变量类型

（1）变量定义、赋值示例程序。

```java
public class Main {
    public static void main(String[] args) {
        int n = 100; // 定义变量n，同时赋值为100
        System.out.println("n = " + n); // 打印n的值

        n = 200; // 变量n赋值为200
        System.out.println("n = " + n); // 打印n的值

        int x = n; // 变量x赋值为n（n的值为200，因此赋值后x的值也是200）
        System.out.println("x = " + x); // 打印x的值

        x = x + 100; // 变量x赋值为x+100（x的值为200，因此赋值后x的值是200+100=300）
        System.out.println("x = " + x); // 打印x的值
        System.out.println("n = " + n); // 再次打印n的值，n应该是200还是300？200
   }
}
```

（2）基本类型：变量是“持有”某个数值

* 整数

  byte(8bit)、short(16bit)、int(32bit)、long(64bit)

* 浮点数

  float(32bit)、double(64bit)

* 字符

  char(16bit)

* 布尔

（3）引用类型：变量是“指向”某个对象

1. 字符串

   String

### 运算

**注意**：由于存在范围限制，如果计算结果超出了范围，就会产生溢出，而溢出*不会出错*，却会得到一个奇怪的结果

（1）整数运算

```java
public class Main {
    public static void main(String[] args) {
        int i = (100 + 200) * (99 - 88); // 3300
        int n = 7 * (5 + (i - 9)); // 23072
        n ++;
        n --;
        System.out.println(i);
        System.out.println(n);
    }
}
```

**注意**：`++n`表示先加1再引用n，`n++`表示先引用n再加1。

* 整数运算的结果永远是精确的；
* 运算结果会自动提升；例如，`short`和`int`计算，结果总是`int`，原因是`short`首先自动被转型为`int`；
* 可以强制转型，但超出范围的强制转型会得到错误的结果；例如，将`int`强制转型为`short`；
* 应该选择合适范围的整型（`int`或`long`），没有必要为了节省内存而使用`byte`和`short`进行整数运算。

* 运算优先级从高到低：
  - `()`
  - `!` `~` `++` `--`
  - `*` `/` `%`
  - `+` `-`
  - `<<` `>>` `>>>`
  - `&` ：与运算，是位运算
  - `|` ：或运算
  - `+=` `-=` `*=` `/=`

（2）浮点数运算

```java
// 浮点数误差
public class Main {
    public static void main(String[] args) {
        double x = 1.0 / 10;
        double y = 1 - 9.0 / 10;
        // 观察x和y是否相等:
        System.out.println(x);
        System.out.println(y);
    }
}
```

* 浮点数常常无法精确表示，并且浮点数的运算结果可能有误差；
* 比较两个浮点数通常比较它们的绝对值之差是否小于一个特定值；
* 整型和浮点型运算时，整型会自动提升为浮点型；
* 可以将浮点型强制转为整型，但超出范围后将始终返回整型的最大值。
* 浮点数运算和整数运算相比，只能进行加减乘除这些数值计算，不能做位运算和移位运算。

（3）布尔运算

* 关系运算符优先级从高到低：
  - `!`
  - `>`，`>=`，`<`，`<=`
  - `==`，`!=`
  - `&&`
  - `||`

* 短路运算：

  ```java
  public class Main {
      public static void main(String[] args) {
          boolean b = 5 < 3;
          boolean result = b && (5 / 0 > 0);
          System.out.println(result);
      }
  }
  ```

  * 因为`false && x`的结果总是`false`，无论`x`是`true`还是`false`，因此，与运算在确定第一个值为`false`后，不再继续计算，而是直接返回`false`。
  * 如果没有短路运算，`&&`后面的表达式会由于除数为`0`而报错，但实际上该语句并未报错，原因在于与运算是短路运算符，提前计算出了结果`false`。

* 三元运算符

  ```java
  public class Main {
      public static void main(String[] args) {
          int n = -100;
          int x = n >= 0 ? n : -n;
          System.out.println(x);
      }
  }
  ```

### 字符和字符串

* Java的字符类型`char`是基本类型，字符串类型`String`是引用类型；

* 基本类型的变量是“持有”某个数值，引用类型的变量是“指向”某个对象；

* 引用类型的变量可以是空值`null`；

* 要区分空值`null`和空字符串`""`。

* 可以使用`+`连接任意字符串和其他数据类型：

  ```java
  public class Main {
      public static void main(String[] args) {
          int age = 25;
          String s = "age is " + age;
          System.out.println(s);
      }
  }
  ```

* 字符串不可变：

  ```java
  public class Main {
      public static void main(String[] args) {
          String s = "hello";
          String t = s;
          s = "world";
          System.out.println(t); // t是"hello"还是"world"? "hello"
      }
  }
  ```

### 数组类型和操作

* 数组类型，数组是引用类型

  **注意**：这里的`s`指向了`name[1]`之后，即将`s`指向了`"XYZ"`，这个是不变的字符串常量

  ```java
  public class Main {
      public static void main(String[] args) {
          String[] names = {"ABC", "XYZ", "zoo"};
          String s = names[1];
          names[1] = "cat";
          System.out.println(s); // s是"XYZ"还是"cat"?  "XYZ" 
      }
  }
  ```
  * 数组元素可以是值类型（如int）或引用类型（如String），但数组本身是引用类型；
  * 数组是同一数据类型的集合，数组一旦创建后，大小就不可变；
  * 可以通过索引访问数组元素，但索引超出范围将报错；

* 数组操作

  * 遍历数组
  
    ```java
    for (int n : ns)
    ```
  
    ```java
    for (int i=0; i<ns.length; i++)
    ```
  
    ```java
    import Java.util.Arrays;
    int[] ns = { 1, 1, 2, 3, 5, 8 };
    System.out.println(Arrays.toString(ns));
    ```
  
  * 数组排序
  
    ```java
    import Java.util.Arrays;
    int[] ns = { 28, 12, 89, 73, 65};
    Arrays.sort(ns); //正序
    ```
  
  * 多维数组
  
    ```java
    import Java.util.Arrays;
    int[][] ns = {
                { 1, 2, 3, 4 },
                { 5, 6, 7, 8 },
                { 9, 10, 11, 12 }
            };
    System.out.println(Arrays.deepToString(ns));
    ```

## 流程控制

### 输入与输出

* 输出：`System.out.println("END");`

* 格式化输出：`System.out.printf("%.4f\n", d);`

* 占位符及说明

  | 占位符 | 说明                             |
  | :----- | :------------------------------- |
  | %d     | 格式化输出整数                   |
  | %x     | 格式化输出十六进制整数           |
  | %f     | 格式化输出浮点数                 |
  | %e     | 格式化输出科学计数法表示的浮点数 |
  | %s     | 格式化字符串                     |

* 输入：

  ```java
  import Java.util.Scanner;
  
  public class Main {
      public static void main(String[] args) {
          Scanner scanner = new Scanner(System.in); // 创建Scanner对象
          System.out.print("Input your name: "); // 打印提示
          String name = scanner.nextLine(); // 读取一行输入并获取字符串
          System.out.print("Input your age: "); // 打印提示
          int age = scanner.nextInt(); // 读取一行输入并获取整数
          System.out.printf("Hi, %s, you are %d\n", name, age); // 格式化输出
      }
  }
  ```

### IF判断

* `if`基本语法：

  ```java
  if (条件) {
      // 条件满足时执行
  } else if (条件) {
      // 
  } else {
      //
  }
  ```

  * 判断浮点数相等时利用差值小于某个临界值来判断：`Math.abs(a-b)<0.00001`
  * 判断引用类型的变量内容是否相等，必须使用`equals()`方法：`s1.equals(s2)`

### SWITCH多重选择

* `switch`语句可以做多重选择，然后执行匹配的`case`语句后续代码；`switch`的计算结果必须是整型、字符串或枚举类型；注意千万不要漏写`break`，建议打开`fall-through`警告；总是写上`default`，建议打开`missing default`警告；**从Java 13开始，`switch`语句升级为表达式，不再需要`break`，并且允许使用`yield`返回值。**

  ```java
  public class Main {
      public static void main(String[] args) {
          int option = 1;
          switch (option) {
          case 1:
              System.out.println("Selected 1");
              break;
          case 2:
              System.out.println("Selected 2");
              break;
          default:
              System.out.println("Selected 3");
              break;
          }
      }
  }
  ```

### WHILE循环

* `while`基本语法：

  ```java
  while (条件表达式) {
      循环语句
  }
  // 继续执行后续代码
  ```

* `do while`基本语法：

  ```java
  do {
      执行循环语句
  } while (条件表达式);
  ```

### FOR循环

* `for`循环基本语法：

  ```java
  for (初始条件; 循环检测条件; 循环后更新计数器) {
      // 执行语句
  }
  ```

### BREAK AND CONTINUE

* `break`：在循环过程中，可以使用`break`语句跳出当前循环
* `continue`：前结束本次循环，直接继续执行下次循环。

# 二、Java进阶

## 面向对象编程基础

### 什么是面向对象？

* 简单来说，人就是一个**对象**，将人身高、年龄等特征封装到人这个对象里，定义人这个类。具体到小明、小芳这些人就被称为**实例**。

* 定义对象

  ```java
  class Person {
      public String name;
      public int age;
  }
  ```

* 创建实例

  ```java
  Person ming = new Person();
  ming.name = "Xiao Ming";
  ```

  * 指向实例的变量`ming`是引用变量

### 方法

* 方法（`method`）可以让外部代码安全地访问实例字段（`filed`）；

  ```java
  public class Main {
      public static void main(String[] args) {
          Person ming = new Person();
          ming.setName("Xiao Ming"); // 设置name
          System.out.println(ming.getName());
      }
  }
  
  class Person {
      private String name;
      
      public String getName() {
          return this.name;
      }
  
      public void setName(String name) {
          this.name = name;
      }
  }
  
  ```

* 方法是一组执行语句，并且可以执行任意逻辑；

* 方法内部遇到return时返回，void表示不返回任何值（注意和返回null不同）；

* 外部代码通过public方法操作实例，内部代码可以调用private方法；其中`this`变量是一个隐含变量，通过`this.field`就可以访问当前实例的字段。

  ```java
  class Person {
      private String name;
      private int birth;
  
      public void setBirth(int birth) {
          this.birth = birth;
      }
  
      public int getAge() {
          return calcAge(2019); // 调用private方法
      }
  
      // private方法:
      private int calcAge(int currentYear) {
          return currentYear - this.birth;
      }
  }
  ```

* 理解方法的参数绑定。

  ```java
  // 基本类型参数绑定
  public class Main {
      public static void main(String[] args) {
          Person p = new Person();
          int n = 15; // n的值为15
          p.setAge(n); // 传入n的值
          System.out.println(p.getAge()); // 15
          n = 20; // n的值改为20
          System.out.println(p.getAge()); // 15还是20?  15
      }
  }
  
  class Person {
      private int age;
  
      public int getAge() {
          return this.age;
      }
  
      public void setAge(int age) {
          this.age = age;
      }
  }
  
  // 引用类型参数绑定
  public class Main {
      public static void main(String[] args) {
          Person p = new Person();
          String[] fullname = new String[] { "Homer", "Simpson" };
          p.setName(fullname); // 传入fullname数组
          System.out.println(p.getName()); // "Homer Simpson"
          fullname[0] = "Bart"; // fullname数组的第一个元素修改为"Bart"
          System.out.println(p.getName()); // "Homer Simpson"还是"Bart Simpson"?  Bart Simpson
      }
  }
  
  class Person {
      private String[] name;
  
      public String getName() {
          return this.name[0] + " " + this.name[1];
      }
  
      public void setName(String[] name) {
          this.name = name;
      }
  }
  ```

  * 基本类型参数的传递，是调用方值的复制。双方各自的后续修改，互不影响。
  * 引用类型参数的传递，调用方的变量，和接收方的参数变量，指向的是同一个对象。双方任意一方对这个对象的修改，都会影响对方（因为指向同一个对象嘛）。

### 构造方法

* 由于构造方法是如此特殊，所以构造方法的名称就是类名。构造方法的参数没有限制，在方法内部，也可以编写任意语句。但是，和普通方法相比，构造方法没有返回值（也没有`void`），调用构造方法，必须用`new`操作符。

  ```java
  public class Main {
      public static void main(String[] args) {
          Person p = new Person("Xiao Ming", 15);
          System.out.println(p.getName());
          System.out.println(p.getAge());
      }
  }
  
  class Person {
      private String name;
      private int age;
  
      // 构造方法
      public Person(String name, int age) {
          this.name = name;
          this.age = age;
      }
      
      public String getName() {
          return this.name;
      }
  
      public int getAge() {
          return this.age;
      }
  }
  ```
  * 如果一个类没有定义构造方法，编译器会自动为我们生成一个默认构造方法，它没有参数，也没有执行语句

  * 如果既要能使用带参数的构造方法，又想保留不带参数的构造方法，那么只能把两个构造方法都定义出来：

  * 既对字段进行初始化，又在构造方法中对字段进行初始化：

    ```java
    class Person {
        private String name = "Unamed";
        private int age = 10;
    
        public Person(String name, int age) {
            this.name = name;
            this.age = age;
        }
    }
    ```

* 多构造方法

  ```java
  class Person {
      private String name;
      private int age;
  
      public Person(String name, int age) {
          this.name = name;
          this.age = age;
      }
  
      public Person(String name) {
          this.name = name;
          this.age = 12;
      }
  
      public Person() {
      }
  }
  ```

  * 在通过`new`操作符调用的时候，编译器通过构造方法的参数数量、位置和类型自动区分

  * 一个构造方法可以调用其他构造方法，这样做的目的是便于代码复用。调用其他构造方法的语法是`this(…)`

    ```java
    class Person {
        private String name;
        private int age;
    
        public Person(String name, int age) {
            this.name = name;
            this.age = age;
        }
    
        public Person(String name) {
            this(name, 18); // 调用另一个构造方法Person(String, int)
        }
    
        public Person() {
            this("Unnamed"); // 调用另一个构造方法Person(String)
        }
    }
    ```

### 方法重载

* 这种方法名相同，但各自的参数不同，称为方法重载（`Overload`）。

  **注意**：方法重载的返回值类型通常都是相同的。

* 重载方法应该完成类似的功能，参考`String`的`indexOf()`；

### 继承

* 继承是面向对象编程中非常强大的一种机制，它首先可以复用代码。当我们让`Student`从`Person`继承时，`Student`就获得了`Person`的所有功能，我们只需要为`Student`编写新增的功能。

  Java使用`extends`关键字来实现继承：

  **注意**：一定要加`super`调用父类的构造方法

  ```java
  public class Main {
      public static void main(String[] args) {
          Student s = new Student("Xiao Ming", 12, 89);
      }
  }
  
  class Person {
      protected String name;
      protected int age;
  
      public Person(String name, int age) {
          this.name = name;
          this.age = age;
      }
  }
  
  // 继承Person
  class Student extends Person {
      protected int score;
  
      public Student(String name, int age, int score) {
          super(name, age); // 调用父类的构造方法Person(String, int)
          this.score = score;
      }
  }
  ```







## Java核心类



## 异常处理



## 反射







# 三、参考书籍

* [廖大神](<https://www.liaoxuefeng.com/wiki/1252599548343744>)

# 四、疑难解答

## 1、使用IntelliJ IDEA 配置Maven

* [参考](<https://blog.csdn.net/qq_32588349/article/details/51461182>)

## 2、Intellij IDEA 打包jar的多种方式

* [参考](<https://blog.csdn.net/Thousa_Ho/article/details/72799871>)

## 3、final关键字解释

* [参考](https://www.cnblogs.com/dolphin0520/p/3736238.html)

* 类：当用final修饰一个类时，表明这个类不能被继承。

* 方法：使用final方法的原因有两个。第一个原因是把方法锁定，以防任何继承类修改它的含义；第二个原因是效率。在早期的Java实现版本中，会将final方法转为内嵌调用。但是如果方法过于庞大，可能看不到内嵌调用带来的任何性能提升。在最近的Java版本中，不需要使用final方法进行这些优化了。

* 变量：对于一个final变量，如果是基本数据类型的变量，则其数值一旦在初始化之后便不能更改；如果是引用类型的变量，则在对其初始化之后便不能再让其指向另一个对象

  