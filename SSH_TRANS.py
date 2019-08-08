"""
script: SSH_TRANS
Author:Ephemeroptera
date:2019-8-6
mail:605686962@qq.com

"""
import utils
import os
import time
import paramiko

# 服务器命令执行
def host_cmd_exe(client,cmd):
    _, stdout, _ = client.exec_command(cmd)
    return stdout.read().decode('utf-8').split()

# 登录信息
host_ip = '218.17.23.152'
port = 22174
username = 'wei.wang'
password = 'wangwei369'

# 文件路径
root_dir = '/home/wei.wang'
src_dir = '/CELEBA_PGGAN/ckpt'
des_dir = './ckpt'

# 已读取文件夹个数
dir_num = 0
ready_trans = False
current_dir = None

# 实例化SSHClient
client = paramiko.SSHClient()

# 自动添加策略，保存服务器的主机名和密钥信息，如果不添加，那么不再本地know_hosts文件中记录的主机将无法连接
# client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# 连接SSH服务端，以用户名和密码进行认证
# client.connect(hostname=host_ip, port=port, username=username, password=password)

# 创建一个通道
transport = paramiko.Transport((host_ip, port))
transport.connect(username=username, password=password)
client._transport = transport

# 获取SFTP实例
sftp = paramiko.SFTPClient.from_transport(transport)

# 扫描buffer区
while (True):
    # 返回buffer区目录列表
    host_dir_list = host_cmd_exe(client, 'cd %s \n ls\n' % (root_dir + src_dir))
    # 检测到目录
    if host_dir_list:
        # 排序
        host_dir_list.sort()
        # 大于两个文件夹开始删除
        if len(host_dir_list)>2:
            # 删除最小项
            print('del old files ..')
            del_dir = host_dir_list[0]
            info = host_cmd_exe(client,'rm -rf %s \n ' % (root_dir + src_dir +'/' + del_dir))
            print(info)
            print('del seccessfully!')
        # 转存最大项
        trans_dir = host_dir_list[-1]
        # 如果为最新目录
        if trans_dir != current_dir:
            print('检测到新文件！！开始处理..')
            current_dir = trans_dir
            ready_trans = True
            # 本地复制目录
            local_dir = os.path.join(des_dir, trans_dir)
            utils.MKDIR(local_dir)
        # 进入目录传送文件
        pwd = root_dir + src_dir + '/' +trans_dir
        trans_files = host_cmd_exe(client, 'cd %s \n ls\n' % pwd)
        # 传输全部子文件
        if 'network.ckpt-17000.data-00000-of-00001' in trans_files and ready_trans:
            for id, filename in enumerate(trans_files):
                sftp.get(pwd + '/' + filename, os.path.join(local_dir, filename))
                print('transfering %s from %s to %s' % (filename, pwd, local_dir))
                # 传输完毕
                if id == len(trans_files) - 1:
                    ready_trans = False
                    print('文件传输完毕！')
                    # 任务结束(删除其他目录并跳出)
                    if 'PG10_level7_False' in host_dir_list:
                        stdin, stdout, _ = client.exec_command('rm -r %s \n rm -r %s \n' % \
                                                               (root_dir + src_dir + '/' + host_dir_list[0],
                                                                root_dir + src_dir + '/' + host_dir_list[1]), get_pty=True)
                        break

    print('~')
    time.sleep(5)# 延时5s

transport.close()

