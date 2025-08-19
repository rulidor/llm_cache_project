
class LRUCache(dict):
    def __init__(self, capacity=100):
        super().__init__()
        self.capacity = capacity
        self.order = []

    def __getitem__(self, key):
        self.order.remove(key)
        self.order.append(key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if key in self:
            self.order.remove(key)
        elif len(self) >= self.capacity:
            oldest = self.order.pop(0)
            del self[oldest]
        self.order.append(key)
        super().__setitem__(key, value)

class SLRUCache:
    def __init__(self, protected_size=70, probationary_size=30):
        self.protected_size = protected_size
        self.probationary_size = probationary_size
        self.protected = {}
        self.probationary = {}
        self.protected_order = []
        self.probationary_order = []
        self.protected_hits = 0
        self.probationary_hits = 0

    def __contains__(self, key):
        return key in self.protected or key in self.probationary

    def __getitem__(self, key):
        if key in self.protected:
            self.protected_order.remove(key)
            self.protected_order.append(key)
            self.protected_hits += 1
            return self.protected[key]
        elif key in self.probationary:
            val = self.probationary.pop(key)
            self.probationary_order.remove(key)
            if len(self.protected) >= self.protected_size:
                evict = self.protected_order.pop(0)
                del self.protected[evict]
            self.protected[key] = val
            self.protected_order.append(key)
            self.probationary_hits += 1
            return val
        else:
            raise KeyError

    def __setitem__(self, key, value):
        if key in self:
            return
        if len(self.probationary) >= self.probationary_size:
            evict = self.probationary_order.pop(0)
            del self.probationary[evict]
        self.probationary[key] = value
        self.probationary_order.append(key)
