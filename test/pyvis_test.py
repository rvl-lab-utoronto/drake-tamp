from pyvis.network import Network

net = Network(directed = True)
net.add_node(1, label = "Node 1") # id, labal
net.add_node(2)
nodes = ["a", "b", "c", "d"]
net.add_nodes(nodes)
print(net.get_node("c"))
net.add_node("abcd", label = "abcd")


net.add_edge("a", "b")
net.add_edge("c", "b")

net.show("mygraph.html")
net.save_graph("test_graph.html")
