#![feature(lazy_cell)]

use std::collections::HashMap;
use std::net::SocketAddr;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, RwLock};
use std::time::Instant;

use bytes::Bytes;
use http::header::{COOKIE, SET_COOKIE};
use http_body_util::{combinators::BoxBody, BodyExt};
use hyper::client::conn::http1::Builder;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response};

use tokio::net::{TcpListener, TcpStream};

mod utils;

use utils::TokioIo;

static COUNTER: AtomicUsize = AtomicUsize::new(8050);
static TABLE: LazyLock<RwLock<HashMap<usize, TimedChild>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

struct TimedChild {
    time: Instant,
    child: Child,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = SocketAddr::from(([127, 0, 0, 1], 8049));

    let listener = TcpListener::bind(addr).await?;
    println!("Listening on http://{}", addr);

    loop {
        let (stream, _) = listener.accept().await?;
        let io = TokioIo::new(stream);

        tokio::task::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .preserve_header_case(true)
                .title_case_headers(true)
                .serve_connection(io, service_fn(proxy))
                .with_upgrades()
                .await
            {
                println!("Failed to serve connection: {:?}", err);
            }
        });
    }
}

/// remove outdated sessions from the process table
fn cleanup() {
    let mut tab = TABLE.write().unwrap();
    // don't worry about small tables
    if tab.len() < 10 {
        return;
    }

    let keys = tab.keys().copied().collect::<Vec<_>>();
    for k in keys {
        if tab.get(&k).unwrap().time.elapsed().as_secs() > 15 * 60 {
            tab.get_mut(&k).unwrap().child.kill().unwrap();
            tab.remove(&k);
        }
    }
}

async fn proxy(
    req: Request<hyper::body::Incoming>,
) -> Result<Response<BoxBody<Bytes, hyper::Error>>, hyper::Error> {
    let mut found_cookie = false;
    let port = match req.headers().get(COOKIE) {
        Some(s) => {
            found_cookie = true;
            String::from_utf8_lossy(s.as_bytes())
                .parse::<usize>()
                .unwrap()
        }
        None => COUNTER.fetch_add(1, Ordering::Relaxed) + 1,
    };
    let host = "127.0.0.1";
    let addr = format!("{}:{}", host, port);

    if !TABLE.read().unwrap().contains_key(&port) {
        let child = Command::new("python")
            .current_dir("../../")
            .arg("board.py")
            .arg("--port")
            .arg(format!("{port}"))
            .spawn()
            .unwrap();
        TABLE.write().unwrap().insert(
            port,
            TimedChild {
                time: std::time::Instant::now(),
                child,
            },
        );
    } else {
        // table contains child, update access time
        let mut tab = TABLE.write().unwrap();
        let tc = tab.get_mut(&port).unwrap();
        tc.time = Instant::now();
    }

    let stream = loop {
        match TcpStream::connect(&addr).await {
            Ok(s) => break s,
            Err(e) => {
                eprintln!("waiting for server to start: {e}");
                cleanup();
                tokio::time::sleep(tokio::time::Duration::from_millis(400))
                    .await;
            }
        }
    };
    let io = TokioIo::new(stream);

    let (mut sender, conn) = Builder::new()
        .preserve_header_case(true)
        .title_case_headers(true)
        .handshake(io)
        .await?;
    tokio::task::spawn(async move {
        if let Err(err) = conn.await {
            println!("Connection failed: {:?}", err);
        }
    });

    let mut resp = sender.send_request(req).await?;
    if !found_cookie {
        resp.headers_mut()
            .insert(SET_COOKIE, COUNTER.load(Ordering::Relaxed).into());
    }
    Ok(resp.map(|b| b.boxed()))
}
