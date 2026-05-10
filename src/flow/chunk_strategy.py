from abc import ABC, abstractmethod

# ==================== 基类 ====================
class ChunkStrategy(ABC):
    """分块策略抽象基类"""
    
    @abstractmethod
    def chunk(self, text: str) -> list:
        """将文本切分成块，返回字符串列表"""
        pass


# ==================== 具体策略：固定大小 ====================
class FixedSizeChunkStrategy(ChunkStrategy):
    """固定大小滑动窗口分块"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        :param chunk_size: 每块最大字符数
        :param overlap: 块间重叠字符数
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must > 0")
        if overlap >= chunk_size:
            raise ValueError("overlap must < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str) -> list:
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunks.append(text[start:end])
            if end == text_len:
                break
            start = end - self.overlap
        return chunks


# ==================== 高级策略（空壳，待实现） ====================
class RecursiveChunkStrategy(ChunkStrategy):
    """递归字符分割（保留段落/句子边界） - 待实现"""
    def chunk(self, text: str) -> list:
        raise NotImplementedError("RecursiveChunkStrategy not implemented yet")


class SemanticChunkStrategy(ChunkStrategy):
    """基于嵌入相似度的语义分块 - 待实现"""
    def chunk(self, text: str) -> list:
        raise NotImplementedError("SemanticChunkStrategy not implemented yet")


class MarkdownHeaderChunkStrategy(ChunkStrategy):
    """按 Markdown 标题层级分块 - 待实现"""
    def chunk(self, text: str) -> list:
        raise NotImplementedError("MarkdownHeaderChunkStrategy not implemented yet")


# ==================== 使用示例 ====================
if __name__ == "__main__":
    sample_text = "人工智能是计算机科学的一个分支。" * 50
    
    # 选择策略（可轻松替换）
    strategy = FixedSizeChunkStrategy(chunk_size=100, overlap=20)
    chunks = strategy.chunk(sample_text)
    
    print(f"共生成 {len(chunks)} 个块")
    for i, block in enumerate(chunks[:3]):
        print(f"块{i+1} (长度{len(block)}): {block[:50]}...")
    
    # 未来切换为递归分割只需一行：
    # strategy = RecursiveChunkStrategy()