class Estimator():
    """
    Initialize the Estimator with the given parameters. Default precision is 16-bit.
    a number of attention heads 
    p pipeline parallel size 
    b microbatch size
    s sequence length 
    h hidden dimension size
    t tensor parallel size 
    L number of transformer layers 
    v vocabulary size
    """
    def __init__(self, model_params=7e9, optimizer='adam', a=32, p=1, b=1, s=512, h=4096, t=1, v=32000, L=32):
        self.model_params=model_params
        self.optimizer=optimizer
        self.a=a
        self.p=p
        self.b=b
        self.s=s
        self.h=h
        self.t=t
        self.v=v 
        self.L=L
    def estimate_gpu_memory_training(self, **params):
        if 'b' in params:
            b=params['b']
        else:
            b=self.b
        if 's' in params:
            s=params['s']
        else:
            s=self.s
        
        if 'stage' in params:
            stage=params['stage']
        else:
            stage=0
        if 'gpu_num' in params:
            gpu_num=params['gpu_num']
        else:
            gpu_num=1

        if stage==1:
            model_memory=(2+2+12/gpu_num)*self.model_params
        elif stage==2:
            model_memory=(2+(2+12)/gpu_num)*self.model_params
        elif stage==3:
            model_memory=((2+2+12)/gpu_num)*self.model_params
        else:
            model_memory=(2+2+12)*self.model_params
        activation_memory=s*b*self.h*(34+5*self.a*s/self.h+4*self.v/(self.L*self.h))*self.L/self.t
        buffer_memory=2*self.model_params
        total_memory=model_memory+activation_memory+buffer_memory
        gb=1024**3
        return total_memory/gb, model_memory/gb, activation_memory/gb, buffer_memory/gb
    
    """
    :precision: 2 for 16-bit, 4 for 32-bit
    """
    def estimate_gpu_memory_inference(self, **params):
        if 'precision' in params:
            precision=params['precision']
        else:
            precision=2
        gb=1024**3
        return precision*self.model_params/gb

