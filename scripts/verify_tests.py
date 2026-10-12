import subprocess,sys,time,xml.etree.ElementTree as ET
from exposurerank.io import ROOT,write_json,revision
start=time.perf_counter();result=subprocess.run([sys.executable,'-m','pytest','--junitxml=evidence/tests.xml'],cwd=ROOT,capture_output=True,text=True)
(ROOT/'evidence/test_log.txt').write_text(result.stdout+result.stderr)
tree=ET.parse(ROOT/'evidence/tests.xml');suite=tree.getroot().find('testsuite');assert suite is not None
record={'tests':int(suite.attrib['tests']),'failures':int(suite.attrib['failures']),'errors':int(suite.attrib['errors']),'skipped':int(suite.attrib['skipped']),'seconds':time.perf_counter()-start,'source_revision':revision(),'exit_code':result.returncode}
write_json(ROOT/'evidence/tests.json',record)
print(result.stdout)
if result.returncode:raise SystemExit(result.returncode)
