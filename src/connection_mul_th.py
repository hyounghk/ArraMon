import socket
import torch
import os
import itertools
import threading, queue
from collections import deque
import multiprocessing

End=b'$'


clients_lock = threading.Lock()

def recvend(conn):

	global End
	total_data=[];data=''
	while True:
		data=conn.recv(8192)
		
		if End in data:
			total_data.append(str(data[:data.find(End)])[2:-1])
			break
		total_data.append(str(data)[2:-1])
		if len(total_data)>1:
			last_pair=total_data[-2]+total_data[-1]
			if str(End) in last_pair:
				total_data[-2]=last_pair[:last_pair.find(End)]
				total_data.pop()
				break

	return ''.join(total_data)

def send_message_to_client(idx, client, message, my_queue): ### You must use the client_lock prior to calling method

	client.send(message)
	data = recvend(client)
	with clients_lock:
		my_queue.put((idx, data))

class connection():
	def __init__(self, ports, sim_num):
		self.my_queue = queue.Queue()			
		self.ports = []
		self.connections = []
		self.clients = []
		active_threads = []
		for port in ports:
			self.ports.append(port)
			print("connectting: " + str(port))
			serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
			serversocket.bind(('localhost', port))
			serversocket.listen(sim_num)
			
			for sim_id in range(sim_num):
				conn, address = serversocket.accept()
				print("waiting: " + str(sim_id))
				self.connections.append(conn)
				print("connected: " + str(sim_id))

	def start(self, cmd_list, epoch=0):
		jobs = []
		for idx, conn in enumerate(self.connections):
			conn.send(cmd_list[idx])
			

		for conn in self.connections:
			buf = self.recv_end(conn)

		return buf

	def recv_end(self, conn):

		global End
		total_data=[];data=''
		while True:
			data=conn.recv(8192)
			
			if End in data:
				total_data.append(str(data[:data.find(End)])[2:-1])
				break
			total_data.append(str(data)[2:-1])
			if len(total_data)>1:
				last_pair=total_data[-2]+total_data[-1]
				if str(End) in last_pair:
					total_data[-2]=last_pair[:last_pair.find(End)]
					total_data.pop()
					break
		return ''.join(total_data)

	def send(self, cmd_list, bsz):

		data = bytearray()
		
		jobs = []
		for idx, conn in enumerate(self.connections):
			th = threading.Thread(target=send_message_to_client, args=(idx, conn, cmd_list[idx], self.my_queue))
			th.start()
			jobs.append(th)


		bufs = []
		for proc in jobs:
			proc.join()
			bufs.append(self.my_queue.get())

		buf_temps = [0 for _ in range(len(bufs))]
		for b in bufs:
			buf_temps[b[0]] = b[1]
		
		buf = []
		for buf_temp in buf_temps:

			buf_temp = buf_temp.split('@')
			buf.extend(buf_temp)


		obj_ids = torch.zeros(bsz)
		obj_pos = torch.zeros(bsz,3)
		agent_pos = torch.zeros(bsz,3)
		agent_rot = torch.zeros(bsz,3)


		img_feats = []
		for i, res_str in enumerate(buf):
			info = res_str.split('#')
			# print(info)
			placed_pos = list(map(float, info[1].strip('()').split(',')))
			a_pos = list(map(float, info[2].strip('()').split(',')))
			a_rot = list(map(float, info[3].strip('()').split(',')))

			obj_ids[i] = int(info[0])
			obj_pos[i,:] = torch.tensor(placed_pos)
			agent_pos[i,:] = torch.tensor(a_pos)
			agent_rot[i,:] = torch.tensor(a_rot)
			img_feats.append(info[4])   

		return obj_ids, obj_pos, agent_pos, agent_rot, img_feats

