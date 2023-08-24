import json
import os


def read_launch_json(src_path):
    lines = open(os.path.join(src_path, ".vscode/launch.json")).readlines()
    lines = [x.strip() for x in lines if len(x.strip()) > 0 and not x.strip().startswith("//")]
    m = json.loads("\n".join(lines))
    return m['configurations']


def get_launch_args(name, src_path):
    m = read_launch_json(src_path=src_path)
    for one in m:
        if one['name'] == name:
            args = one.get('args', [])
            args = [x if " " not in x else "'" + x + "'" for x in args]
            return args
    return None


def get_args_kv(args):
    m = {}
    i = 0
    while i < len(args):
        k = args[i].strip()
        if k.startswith("--"):
            offset = k.find("=")
            if offset > 0:
                k1 = k[0: offset].strip()
                v1 = k[offset + 1:].strip()
                assert len(k1) > 0
                assert len(v1) > 0
                assert k1 not in m
                m.update({k1: v1})
                i += 1
                continue
            v = args[i+1].strip()
            if not v.startswith("--"):
                i += 2
            else:
                v = ""
                i += 1
            assert k not in m
            m.update({k: v})
        else:
            i += 1
    return m


def get_program_config(name, src_path):
    args = get_launch_args(name, src_path)
    m = get_args_kv(args)
    return args, m
