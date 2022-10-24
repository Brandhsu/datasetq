def decorate_with_indices(cls):
    """Decorates a dataset to return its index
    
    Example:
        from torchvision.datasets import MNIST
        from datasetq.dataset import decorate_with_indices
        
        MNISTWithIndices = decorate_with_indices(MNIST)
        dataset = MNISTWithIndices(root, train=True)
    """

    def __getitem__(self, index:int):
        return index, cls.__getitem__(self, index)

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })
