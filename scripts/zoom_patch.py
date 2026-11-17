"""Notebook code that writes SRC/compact_infer_zoom.py: heliosli's v4c worker with the zoom-specialist input transform
(every local view, ids 1-5, shows its centre 1/ZOOM resized back to the cache size, before the centre crop) -- the same
transform build_b_train.py --zoom applies in the trainer.  worker_cell(zoom) returns the code; it defines ZOOM_WORKER."""


def worker_cell(zoom, indent=""):
    code = r'''ZOOM = %r
_zw = (SRC / 'compact_infer.py').read_text(encoding='utf-8')
_ZDEF = (chr(10) + 'ZOOM = ' + repr(ZOOM) + chr(10) + chr(10) + chr(10) + 'def zoom_locals(img, views, z):' + chr(10)
         + '    import torch.nn.functional as F' + chr(10)
         + '    sel = np.flatnonzero((views >= 1) & (views <= 5))' + chr(10)
         + '    if z <= 1 or not len(sel):' + chr(10) + '        return img' + chr(10)
         + '    S = img.shape[-1]' + chr(10) + '    c = int(round(S / z))' + chr(10) + '    o = (S - c) // 2' + chr(10)
         + '    x = torch.from_numpy(np.ascontiguousarray(img[sel, :, o:o + c, o:o + c])).float()' + chr(10)
         + "    x = F.interpolate(x, size=(S, S), mode='bilinear', align_corners=False)" + chr(10)
         + '    img = np.array(img, copy=True)' + chr(10)
         + '    img[sel] = x.round().clamp(0, 255).to(torch.uint8).numpy()' + chr(10) + '    return img' + chr(10))
_A1 = chr(10) + 'import torch' + chr(10)
_A2 = '            m = meta[rows[:, 1]]' + chr(10)
assert _zw.count(_A1) == 1 and _zw.count(_A2) == 1
_zw = _zw.replace(_A1, _A1 + _ZDEF, 1).replace(_A2, _A2 + '            img = zoom_locals(img, m[:, 3], ZOOM)' + chr(10), 1)
(SRC / 'compact_infer_zoom.py').write_text(_zw, encoding='utf-8')
ZOOM_WORKER = 'compact_infer_zoom.py'
print('ZOOM worker written, zoom', ZOOM, flush=True)
''' % zoom
    return "".join(indent + line if line.strip() else line for line in code.splitlines(keepends=True))
