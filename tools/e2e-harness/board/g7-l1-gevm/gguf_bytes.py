import struct
# ggml quant enum -> (block_bytes, block_nweights)
BLK={0:(4,1),1:(2,1),8:(34,32),10:(84,256),11:(110,256),12:(144,256),13:(176,256),14:(210,256)}
TN={0:"f32",1:"f16",8:"q8_0",10:"q2_K",11:"q3_K",12:"q4_K",13:"q5_K",14:"q6_K"}
f=open("/data/tinyllama-1.1b-Q5_K_M.gguf","rb")
assert f.read(4)==b"GGUF"; ver=struct.unpack("<I",f.read(4))[0]
nt=struct.unpack("<Q",f.read(8))[0]; nkv=struct.unpack("<Q",f.read(8))[0]
def rstr():
    n=struct.unpack("<Q",f.read(8))[0]; return f.read(n)
def skipval(t):
    if t in (0,1,7): f.read(1)
    elif t in (2,3): f.read(2)
    elif t in (4,5,6): f.read(4)
    elif t in (10,11,12): f.read(8)
    elif t==8: rstr()
    elif t==9:
        et=struct.unpack("<I",f.read(4))[0]; n=struct.unpack("<Q",f.read(8))[0]
        for _ in range(n): skipval(et)
for _ in range(nkv):
    rstr(); vt=struct.unpack("<I",f.read(4))[0]; skipval(vt)
tot=0; embd=0; out=0; byt={}
for _ in range(nt):
    name=rstr().decode(); nd=struct.unpack("<I",f.read(4))[0]
    dims=[struct.unpack("<Q",f.read(8))[0] for _ in range(nd)]
    ty=struct.unpack("<I",f.read(4))[0]; off=struct.unpack("<Q",f.read(8))[0]
    ne=1
    for d in dims: ne*=d
    bb,bn=BLK.get(ty,(0,1)); nb=ne//bn if bn else 0; b=nb*bb
    tot+=b; k=TN.get(ty,str(ty)); byt[k]=byt.get(k,0)+b
    if "token_embd" in name: embd=b
    if name.startswith("output.") or name=="output.weight": out=b
print("gguf_ver",ver,"n_tensors",nt,"nkv",nkv)
print("total_tensor_bytes = %.2f MiB (%.4f GiB)"%(tot/2**20,tot/2**30))
print("token_embd = %.2f MiB   output(lm_head) = %.2f MiB"%(embd/2**20,out/2**20))
print("per-token decode floor (total - token_embd) = %.2f MiB (%.4f GiB)"%((tot-embd)/2**20,(tot-embd)/2**30))
print("bytes_by_quant_type:", {k:"%.1fMiB"%(v/2**20) for k,v in sorted(byt.items())})
