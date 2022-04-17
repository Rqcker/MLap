import subprocess
import json
#variable 'out' is subprocess output info
top_info = subprocess.Popen(["top", "-n", "1"], stdout=subprocess.PIPE)
out, err = top_info.communicate()

out_info = out.decode('unicode-escape')
print(out_info)

output=json.dumps(out_info)
f2=open('output.json','w')
f2.write(output)
f2.close()

lines = []
lines = out_info.split('\n')
