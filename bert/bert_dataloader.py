import json

from torch.utils.data import Dataset

class JsonlDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_length):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        # Store the byte offset of each line for random access
        self.line_offsets = []
        with open(file_path, "r", encoding="utf-8") as f:
            offset = 0
            for line in f:
                self.line_offsets.append(offset)
                offset += len(line.encode("utf-8"))

    def __len__(self):
        return len(self.line_offsets)

    def __getitem__(self, idx):
        offset = self.line_offsets[idx]
        with open(self.file_path, "r", encoding="utf-8") as f:
            f.seek(offset)
            line = f.readline()
            data = json.loads(line)
            text = data["text"]
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
            # Remove batch dimension
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}
            return encoding
