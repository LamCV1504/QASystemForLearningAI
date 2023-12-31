import pyodbc
from predict_label import load_data
connstr = 'DRIVER={ODBC Driver 17 for SQL Server}; SERVER=DESKTOP-MPU5E0P; Database=DATASET; UID=sa; PWD=123;Encrypt connection=false;'

if __name__ == '__main__':
    conx = pyodbc.connect(connstr)
    cursor = conx.cursor()
    data = load_data([])
    cursor.execute("if exists(select * from information_schema.tables where table_schema = 'dbo'  AND Table_Name ='data') begin drop table data end")
    cursor.execute("create table data(id_key int identity(1,1) primary key ,text ntext not null, label1 nvarchar(250) not null,label2 nvarchar(250) not null)")
    for row in data:
        cursor.execute("insert data values(N'"+row[0]+"','" + row[1]+"','" + row[2] + "')") 
    print("Done!")
    cursor.commit()
    conx.close()