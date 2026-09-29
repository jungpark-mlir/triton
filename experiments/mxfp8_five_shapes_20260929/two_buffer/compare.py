"""Alternating output-store comparison on the same rotating input addresses."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path('/home/jungpark/mnt/gemm/triton')
sys.path.insert(0,str(ROOT/'third_party/amd/python/examples/gluon/gfx1250_gemm'))
import kernels_mxfp8_five_shapes_0929 as k
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--indices',type=int,nargs='*',default=list(range(12)))
    p.add_argument('--rounds',type=int,default=10)
    p.add_argument('--output',default=str(Path(__file__).with_name('paired.json')))
    args=p.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    out=Path(args.output).parent
    results=[]
    for index in args.indices:
        shape=k.SHAPES[index]
        print('START',index,shape,flush=True)
        tensors,sa,sb=k.make_inputs(*shape[:3],shape[4]==256,42)
        launchers={mode:k.launcher(shape,mode) for mode in ['serial','double']}
        resource={}
        for mode,launch in launchers.items():
            compiled=launch(tensors)
            torch.cuda.synchronize()
            err=k.check(tensors,sa,sb)
            resource[mode]=dict(vgpr=compiled.n_regs,spills=compiled.n_spills,lds=compiled.metadata.shared,max_relative_error=err)
            (out/f'{index}_{mode}.amdgcn').write_text(compiled.asm['amdgcn'])
            print('CHECK',index,mode,resource[mode],flush=True)
        del sa,sb,compiled
        torch.cuda.empty_cache()
        free,_=torch.cuda.mem_get_info()
        nbytes=sum(t.numel()*t.element_size() for t in tensors)
        copies=max(1,min(100,1+int(max(0,free-max(2*1024**3,free//2))//nbytes)))
        pool=[tensors]+[tuple(t.clone() for t in tensors) for _ in range(copies-1)]
        graphs={}
        for mode,launch in launchers.items():
            for i in range(20): launch(pool[i%copies])
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for i in range(100): launch(pool[i%copies])
            graphs[mode]=graph
            for _ in range(3): graph.replay()
        torch.cuda.synchronize()
        times={mode:[] for mode in graphs}
        for r in range(args.rounds):
            for mode in (['serial','double'] if r%2==0 else ['double','serial']):
                start=torch.cuda.Event(enable_timing=True)
                end=torch.cuda.Event(enable_timing=True)
                start.record();graphs[mode].replay();end.record();end.synchronize()
                times[mode].append(start.elapsed_time(end)*10)
        med={mode:statistics.median(values) for mode,values in times.items()}
        wins=sum(b<a for a,b in zip(times['serial'],times['double']))
        row=dict(index=index,shape=shape[:3],cta=[shape[3],shape[3],shape[4]],cluster=shape[5],copies=copies,resources=resource,times_us=times,median_us=med,double_wins=wins,rounds=args.rounds,reduction_pct=(1-med['double']/med['serial'])*100)
        results.append(row)
        print('RESULT',json.dumps(row),flush=True)
        Path(args.output).write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(k.__file__).read_bytes()).hexdigest(),results=results),indent=2)+'\n')
        del graphs,graph,pool,tensors,launchers,launch
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__': main()
