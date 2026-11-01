#!/usr/bin/env python3
"""Download a checksum-pinned public release and compile Core ML offline."""
import argparse, hashlib, json, plistlib, shutil, subprocess, tempfile, urllib.request, zipfile
from pathlib import Path, PurePosixPath
ROOT = Path(__file__).resolve().parent.parent

def prepare(archive, destination, platform="macOS"):
    manifest = json.loads((ROOT / 'model-manifest.json').read_text())
    with tempfile.TemporaryDirectory(prefix='image-safety-models-') as scratch:
        scratch = Path(scratch)
        with zipfile.ZipFile(archive) as z:
            allowed = set(manifest['sha256']) | {'manifest.json'}
            if set(z.namelist()) != allowed or len(z.namelist()) != len(allowed):
                raise ValueError('Unexpected or duplicate release members')
            for item in z.infolist():
                path = PurePosixPath(item.filename)
                if path.is_absolute() or '..' in path.parts or (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Unsafe archive member')
                if item.file_size > 50 * 1024 * 1024:
                    raise ValueError('Oversized model member')
                target = scratch / path
                target.parent.mkdir(parents=True, exist_ok=True)
                data = z.read(item)
                if item.filename != 'manifest.json' and hashlib.sha256(data).hexdigest() != manifest['sha256'][item.filename]:
                    raise ValueError('Model checksum mismatch: ' + item.filename)
                target.write_bytes(data)
        if json.loads((scratch / 'manifest.json').read_text()) != manifest:
            raise ValueError('Release manifest mismatch')
        if destination.exists():
            raise ValueError('Output exists; choose a new --output directory')
        destination.mkdir(parents=True)
        (destination / 'Info.plist').write_bytes(plistlib.dumps({'CFBundleIdentifier':'local.imagesafety.models','CFBundlePackageType':'BNDL','CFBundleVersion':'1'}))
        try:
            for model in manifest['models']:
                subprocess.run(['xcrun','coremlcompiler','compile',str(scratch / model),str(destination),'--platform',platform,'--deployment-target','17.0' if platform == 'iOS' else '14.0'], check=True)
            for name in manifest['sha256']:
                if not any(name.startswith(model + '/') for model in manifest['models']):
                    target = destination / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(scratch / name, target)
            shutil.copyfile(scratch / 'manifest.json', destination / 'model-manifest.json')
        except BaseException:
            # Keep compiler diagnostics and partial output for inspection. Never accept it as a valid bundle.
            (destination / 'INCOMPLETE').write_text('Compilation did not complete.\n')
            raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use an already downloaded v1.0.0 model ZIP')
    parser.add_argument('--platform', choices=['macOS', 'iOS'], default='macOS')
    parser.add_argument('--output', type=Path, default=ROOT / 'build/ImageSafetyModels.bundle')
    args = parser.parse_args()
    if args.archive:
        prepare(args.archive, args.output.resolve(), args.platform)
    else:
        release = json.loads((ROOT / 'release-models.json').read_text())
        with tempfile.TemporaryDirectory(prefix='image-safety-download-') as scratch:
            archive = Path(scratch) / 'models.zip'
            req = urllib.request.Request(release['url'], headers={'User-Agent':'image-safety-coreml/1.0'})
            size = 0
            digest = hashlib.sha256()
            with urllib.request.urlopen(req, timeout=120) as response, archive.open('wb') as out:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > release['bytes']: raise ValueError('Oversized release download')
                    digest.update(chunk);out.write(chunk)
            if size != release['bytes'] or digest.hexdigest() != release['sha256']:
                raise ValueError('Release checksum mismatch')
            prepare(archive, args.output.resolve(), args.platform)
    print('Models ready:', args.output)

if __name__ == '__main__': main()
