import pandas as pd

from realnetwork import path

graph_dict ={"Nodes" : "" , "Edges" : ""}

from_node = []
to_node = []
df = pd.DataFrame()

graph_name ="amazon"


# def get_data_frame():
#     tmp = 0
#     node = False
#     edge = False
#     start = False
#
#     with open("com-amazon.ungraph.txt") as f:
#         for line in f:
#             for word in line.split():
#                 if node == True:
#                     graph_dict['Nodes'] = word
#                     node = False
#                 if edge == True:
#                     graph_dict['Edges'] = word
#                     edge = False
#
#                 if start == True:
#                     if tmp % 2 == 0:
#                         from_node.append(word)
#                     else:
#                         to_node.append(word)
#                 if word == "Nodes:":
#                     node = True
#                 if word =="Edges:":
#                     edge = True
#
#                 if word == "ToNodeId":
#                     start = True
#                 tmp += 1
#
#         df['FromNodeId'] = from_node
#         df['ToNodeId'] = to_node
#         df['Nodes'] = graph_dict["Nodes"]
#         df['Edges'] = graph_dict["Edges"]
#
# 	#you might wanna edit this path to save the file to your preferred location
#     df.to_csv(r"D:\Desktop\amazon.csv")

def get_data_frame(graph_name):
    graph_dict = {"Nodes": "", "Edges": ""}

    from_node = []
    to_node = []
    df = pd.DataFrame()

    # graph_name = "amazon"

    tmp = 0
    node = False
    edge = False
    start = False

    with open(path.CSV_NETWORK_DIR_PATH + "\\" + graph_name + ".txt", "r") as f:
        for line in f:
            for word in line.split():
                if node == True:
                    graph_dict['Nodes'] = word
                    node = False
                if edge == True:
                    graph_dict['Edges'] = word
                    edge = False

                if start == True:
                    if tmp % 2 == 0:
                        from_node.append(word)
                    else:
                        to_node.append(word)
                if word == "Nodes:":
                    node = True
                if word == "Edges:":
                    edge = True

                if word == "ToNodeId":
                    start = True
                tmp += 1

        df['FromNodeId'] = from_node
        df['ToNodeId'] = to_node
        df['Nodes'] = graph_dict["Nodes"]
        df['Edges'] = graph_dict["Edges"]

    # you might wanna edit this path to save the file to your preferred location
    # df.to_csv(r"csv\amazon.csv")
    df.to_csv(path.CSV_NETWORK_DIR_PATH + "\\" + graph_name + ".csv")

#get_data_frame()


