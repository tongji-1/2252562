import pymysql

# 连接mysql
database = pymysql.connect(user='root',password='',host='localhost', charset='utf8mb4')

# 查看连接状态
cursor = database.cursor()

# 读取 sql 文件
with open('flooddata.sql', 'r', encoding='utf-8') as file:
    sql_commands = file.read()

# 分割每条 SQL 语句
commands = sql_commands.split(';')

# 执行每条命令
for command in commands:
    command = command.strip()  # 去掉首尾空格
    if command:  # 忽略空行
        try:
            cursor.execute(command)
        except pymysql.MySQLError as e:
            print(f"Error executing command: {command}\nError: {e}")

# 提交更改并关闭连接
database.commit()
cursor.close()
database.close()
