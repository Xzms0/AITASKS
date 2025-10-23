# Python 学习笔记（纠正与补充）

> 102501316 — 改进版

此文档在保留原作者笔记风格的基础上，对部分措辞进行澄清、补充更准确的概念说明，并给出可运行示例，便于复习与教学。

---

## 数据结构：List 与 Dict

- List（列表）和 Dict（字典）是 Python 中最常用的数据结构之一，支持嵌套并提供许多内置方法，使用方便。

### 注意事项与补充
- `dict.keys()`、`dict.values()` 返回的是视图对象（`dict_keys`、`dict_values`），它们是可迭代的并且会随字典变化而动态更新。只有当你需要对键或值进行索引或固定快照时，才需要使用 `list(d.keys())`。

```python
d = {'a': 1, 'b': 2}
print(type(d.keys()))  # <class 'dict_keys'>
for k in d.keys():
    print(k)
# 如果想得到可以索引的静态列表：
keys_list = list(d.keys())
```

- `list.sort()` 是原地排序（in-place），会修改原列表并返回 `None`。因此不要用 `list1.sort() == list2.sort()` 来判断两个列表是否相等。正确的方法是使用 `sorted()` （返回新列表）：

```python
a = [3,1,2]
b = [2,3,1]
print(sorted(a) == sorted(b))  # True
# 或者复制并原地排序：
a_copy = a[:]
a_copy.sort()
```

---

## Lambda（匿名函数）

- Lambda 表达式用于定义单行匿名函数，语法上只允许单个表达式，因此不适合包含多条语句或复杂逻辑，但在需要一次性小函数时非常有用（例如作为 `key=` 参数或与 `map`/`filter` 一起使用）。

```python
# 示例：按年龄排序
users = [{'name':'a','age':30},{'name':'b','age':20}]
users_sorted = sorted(users, key=lambda u: u['age'])
```

- 常用替代：`operator.itemgetter` 或 `operator.attrgetter`，在某些场景更高效且可读性更好。

---

## 装饰器（Decorator）

- 装饰器用于给函数或方法在不修改其源码的情况下添加行为，例如日志、缓存、权限校验等。常见形式包括无参装饰器与带参装饰器（装饰器工厂）。

- 使用 `functools.wraps` 来保持被装饰函数的元信息（`__name__`, `__doc__`）。

```python
import functools
import time

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end-start:.6f}s")
        return result
    return wrapper

@timer
def work(n):
    sum(i*i for i in range(n))
```

---

## 类与魔术方法（Magic Methods）

- 类是封装数据与行为的工具。常见魔术方法有 `__init__`, `__repr__`, `__str__`, `__len__`, `__eq__` 等。
- 实现 `__eq__` 时尽量不要修改对象状态（例如不要在比较函数中对属性排序），应返回比较结果。下面示例为合理实现：

```python
class MyZoo:
    def __init__(self, animals=None):
        self.animals = dict(animals) if animals else {}
    def __eq__(self, other):
        if not isinstance(other, MyZoo):
            return NotImplemented
        return set(self.animals.keys()) == set(other.animals.keys())
```

---

## 正则表达式（re）

- `re` 模块提供正则表达式匹配、搜索、替换等功能。常用方法包括 `re.match`, `re.search`, `re.findall`, `re.sub`。

```python
import re
s = "2025-10-23"
m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
if m:
    year, month, day = m.groups()
```

---

## 列表推导式

- 列表推导式是生成列表的简洁语法，也可以生成集合（set）与字典（dict）。支持条件和多重循环，功能比 `map()` 更强。示例：

```python
squares = [x*x for x in range(10) if x % 2 == 0]
pairs = [(i,j) for i in range(3) for j in range(2)]
```

---

## 生成器（generator）与 `yield` 关键字

- 生成器是实现迭代器协议的一种对象，通常由包含 `yield` 的函数创建。生成器的主要特点是按需生成数据，节省内存（lazy evaluation）。

- 生成器是迭代器的一种：所有生成器都是迭代器，但并非所有迭代器都是生成器（迭代器也可以是自定义类）。

```python
# 生成器示例：斐波那契
def fib():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

import itertools
for x in itertools.islice(fib(), 10):
    print(x)
```

- 高级用法包括：`send()`（向生成器发送值）、`throw()`（向生成器抛出异常）、`close()`（关闭生成器）。

---

## 面向对象（OOP）与 Type Hint

- Type Hint 可以提高代码可读性并支持静态类型检查（如 mypy），建议在项目中逐步引入。
- `from __future__ import annotations` 可以延迟注解解析，便于引用类本身作为类型。示例：

```python
from __future__ import annotations
class Node:
    def __init__(self, next: 'Node' | None = None):
        self.next = next
```

---

## 常见陷阱（汇总）

1. 可变默认参数（`def f(x, lst=[])`）
2. `list.sort()` 返回 None
3. 延迟绑定闭包（loop variables in lambdas）
4. `is` 与 `==` 的区别（`is` 比较对象标识，`==` 比较值）
5. 浅拷贝与深拷贝（`copy.copy` vs `copy.deepcopy`）

---

## 建议的下一步
- 将该纠正文档与原文并列保留，逐节对照练习代码片段以巩固理解。
- 我可以将此文档加入到仓库（已创建），并根据需要将每个示例拆分成单独的 `.py` 文件便于运行与测试。

---

（文档已创建为 `task1/Documents/Python_correct.md`）