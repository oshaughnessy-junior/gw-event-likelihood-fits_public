"""Validate a versioned public DCC archive/member retrieval witness."""
import hashlib
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import urljoin


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs=[]

    def handle_starttag(self,tag,attrs):
        if tag=='a':
            self.hrefs.extend(v for k,v in attrs if k=='href')


def require_dcc_witness(root,proof,metadata):
    root=Path(root).resolve()
    event=proof['event']
    if not re.fullmatch(r'GW\d{6}_\d{6}',event):
        raise ValueError('DCC event identity is malformed')
    if proof['release_id']!='P2000223' or proof['release_version']!='v7':
        raise ValueError('Only the audited GWTC-2 DCC v7 release is supported')
    archive='https://dcc.ligo.org/public/0169/P2000223/007/'+event+'.tar'
    page='https://dcc.ligo.org/LIGO-P2000223-v7/public'
    if proof['public_file_url']!=archive or proof['public_file_name']!=event+'.tar':
        raise ValueError('DCC archive URL or filename differs from its pinned release')
    if metadata.get('schema')!='dcc-public-archive-witness-v1' or metadata.get('event')!=event:
        raise ValueError('DCC retrieval witness has another schema/event')
    expected=dict(release_id='P2000223',release_version='v7',landing_page_url=page,
        archive_url=archive,access='public_https_without_authentication',
        checksum_authority='observed_https_retrieval')
    if any(metadata.get(k)!=v for k,v in expected.items()):
        raise ValueError('DCC witness release/access differs')
    checksum=dict(algorithm='sha256',value=metadata.get('archive_sha256'))
    if not re.fullmatch('[0-9a-f]{64}',str(checksum['value'])):
        raise ValueError('DCC archive digest is malformed')
    if proof['public_file_checksum']!=checksum or proof.get('observed_file_checksum')!=checksum or proof['public_file_sha256']!=checksum['value']:
        raise ValueError('DCC archive checksum differs from its retrieval witness')
    member=metadata.get('archive_member')
    if member!=event+'/'+event+'.h5' or proof.get('archive_member')!=member:
        raise ValueError('DCC archive member differs from the audited layout')
    digest=metadata.get('member_sha256')
    if not re.fullmatch('[0-9a-f]{64}',str(digest)):
        raise ValueError('DCC member digest is malformed')
    if proof.get('archive_member_sha256')!=digest or proof['source_file_sha256']!=digest:
        raise ValueError('DCC member differs from the fit source')
    if proof.get('posterior_group')!=metadata.get('posterior_group'):
        raise ValueError('DCC posterior group differs from the retrieval witness')
    if not metadata.get('retrieved_at') or not re.fullmatch('[0-9a-f]{64}',str(metadata.get('retriever_sha256'))):
        raise ValueError('DCC witness lacks its retrieval provenance')
    ref=metadata.get('landing_page',{});relative=Path(ref.get('path',''))
    if relative.is_absolute()or'..'in relative.parts:
        raise ValueError('DCC landing page path must be contained')
    path=(root/relative).resolve()
    if not path.is_relative_to(root)or not path.is_file():
        raise ValueError('DCC landing page snapshot is missing')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=ref.get('sha256'):
        raise ValueError('DCC landing page snapshot checksum differs')
    links=Links();links.feed(raw.decode('utf-8'))
    if archive not in [urljoin(page,href)for href in links.hrefs]:
        raise ValueError('DCC landing page does not list this archive')
    if not re.search(r'P2000223-v7',raw.decode('utf-8')):
        raise ValueError('DCC landing page names another document version')
    return metadata
