from dlgo.data.processor import GoDataProcessor

go = GoDataProcessor()
data = go.load_go_data('train', 100)

print(data)
