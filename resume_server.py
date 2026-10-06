"""FastMCP server: export live resume plain text from a Google Doc via service account."""

from __future__ import annotations

import io
import os

from fastmcp import FastMCP
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

mcp = FastMCP("resume")

DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"


def _credentials_path() -> str:
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get(
        "SERVICE_ACCOUNT_PATH", "./service_account.json"
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Service account key not found at {path!r}. "
            "Set SERVICE_ACCOUNT_PATH or GOOGLE_APPLICATION_CREDENTIALS."
        )
    return path


def _google_credentials():
    creds = service_account.Credentials.from_service_account_file(
        _credentials_path(),
        scopes=[DRIVE_SCOPE],
    )
    subject = os.environ.get("GOOGLE_SUBJECT_EMAIL")
    if subject:
        creds = creds.with_subject(subject)
    return creds


def _drive_service():
    return build("drive", "v3", credentials=_google_credentials(), cache_discovery=False)


def _doc_id() -> str:
    doc_id = os.environ.get("GOOGLE_DOC_ID", "").strip()
    if not doc_id:
        raise ValueError(
            "GOOGLE_DOC_ID is not set. Pass it in the server environment when starting the agent."
        )
    return doc_id


def _export_plain_text(file_id: str) -> str:
    service = _drive_service()
    request = service.files().export_media(
        fileId=file_id,
        mimeType="text/plain",
    )
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue().decode("utf-8")


@mcp.tool
def get_resume_text() -> str:
    """Return the full plain-text content of the configured Google Doc resume."""
    try:
        return _export_plain_text(_doc_id())
    except HttpError as err:
        status = err.resp.status if err.resp else "unknown"
        if status == 403:
            raise RuntimeError(
                "Drive API returned 403. Share the Google Doc with the service account "
                "client_email from your JSON key, or check domain-wide delegation."
            ) from err
        if status == 404:
            raise RuntimeError(
                "Document not found. Check GOOGLE_DOC_ID (ID from /document/d/{ID}/edit)."
            ) from err
        raise RuntimeError(f"Google Drive export failed (HTTP {status}): {err}") from err
    except (FileNotFoundError, ValueError) as err:
        raise RuntimeError(str(err)) from err


@mcp.tool
def get_document_info() -> str:
    """Return resume document metadata (name, modified time, MIME type)."""
    try:
        service = _drive_service()
        meta = (
            service.files()
            .get(
                fileId=_doc_id(),
                fields="id,name,mimeType,modifiedTime,webViewLink",
            )
            .execute()
        )
        lines = [
            f"name: {meta.get('name', '')}",
            f"id: {meta.get('id', '')}",
            f"mimeType: {meta.get('mimeType', '')}",
            f"modifiedTime: {meta.get('modifiedTime', '')}",
            f"webViewLink: {meta.get('webViewLink', '')}",
        ]
        return "\n".join(lines)
    except HttpError as err:
        status = err.resp.status if err.resp else "unknown"
        raise RuntimeError(f"Google Drive files.get failed (HTTP {status}): {err}") from err
    except (FileNotFoundError, ValueError) as err:
        raise RuntimeError(str(err)) from err


if __name__ == "__main__":
    mcp.run()
