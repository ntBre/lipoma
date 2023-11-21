#![feature(lazy_cell)]

use std::collections::HashMap;
use std::net::SocketAddr;
use std::os::unix::process::CommandExt;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, RwLock};
use std::time::Instant;

use bytes::Bytes;
use http::header::{COOKIE, SET_COOKIE};
use http_body_util::{
    Full,
    {combinators::BoxBody, BodyExt},
};
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

#[derive(Clone, Copy, PartialEq)]
enum PType {
    Board,
    Twod,
}

impl PType {
    fn as_str(&self) -> &'static str {
        match self {
            PType::Board => "board.py",
            PType::Twod => "twod.py",
        }
    }
}

struct TimedChild {
    time: Instant,
    child: Child,
    typ: PType,
}

impl TimedChild {
    fn kill(self) {
        let pid = self.child.id();
        Command::new("pkill")
            .arg("-g")
            .arg(format!("{pid}"))
            .output()
            .unwrap();
    }
}

use clap::Parser;

#[derive(Parser)]
struct Cli {
    /// Sets a custom config file
    #[arg(short, long, default_value_t = String::from("127.0.0.1:8048"))]
    addr: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let addr: SocketAddr = cli.addr.parse().unwrap();

    let listener = TcpListener::bind(addr).await?;
    println!("Listening on http://{}", addr);

    loop {
        let (stream, _) = listener.accept().await?;
        let io = TokioIo::new(stream);

        tokio::task::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .preserve_header_case(true)
                .title_case_headers(true)
                .serve_connection(io, service_fn(dispatch))
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
    eprintln!("calling cleanup with {} entries", tab.len());
    let keys = tab.keys().copied().collect::<Vec<_>>();
    eprintln!("found {} keys: {:?}", keys.len(), keys);
    for k in keys {
        if tab.get(&k).unwrap().time.elapsed().as_secs() > 5 * 60 {
            let child = tab.remove(&k).unwrap();
            child.kill();
            eprintln!("removing instance on {k}");
        }
    }
}

async fn index(
    _: Request<hyper::body::Incoming>,
) -> Result<Response<BoxBody<Bytes, hyper::Error>>, hyper::Error> {
    Ok(Response::new(
        Full::new(Bytes::from(
            std::fs::read_to_string("proxy/index.html").unwrap(),
        ))
        .map_err(|never| match never {})
        .boxed(),
    ))
}

async fn dispatch(
    req: Request<hyper::body::Incoming>,
) -> Result<Response<BoxBody<Bytes, hyper::Error>>, hyper::Error> {
    if req.uri() == "/" {
        index(req).await
    } else {
        let cookie = check_cookie(&req);
        let port = match cookie {
            Ok(c) | Err(c) => c,
        };
        if req.uri() == "/board" {
            eprintln!("got board");
            proxy(req, cookie, port, Some(PType::Board)).await
        } else if req.uri() == "/twod" {
            eprintln!("got twod",);
            proxy(req, cookie, port, Some(PType::Twod)).await
        } else {
            proxy(req, cookie, port, None).await
        }
    }
}

async fn proxy(
    req: Request<hyper::body::Incoming>,
    cookie: Result<usize, usize>,
    port: usize,
    cmd: Option<PType>,
) -> Result<Response<BoxBody<Bytes, hyper::Error>>, hyper::Error> {
    let host = "127.0.0.1";
    let addr = format!("{}:{}", host, port);

    if !TABLE.read().unwrap().contains_key(&port) {
        // new value, just start up the cmd
        if let Some(value) = start_cmd(cmd, port) {
            return value;
        }
    } else if TABLE
        .read()
        .unwrap()
        .get(&port)
        .is_some_and(|c| cmd.is_some_and(|cmd| cmd != c.typ))
    {
        // table contains a different dashboard than is now requested, so kill
        // the existing child process and start a new one
        {
            let mut tab = TABLE.write().unwrap();
            let child = tab.remove(&port).unwrap();
            child.kill();
        }
        if let Some(value) = start_cmd(cmd, port) {
            return value;
        }
    } else {
        // using existing child, update access time
        let mut tab = TABLE.write().unwrap();
        let tc = tab.get_mut(&port).unwrap();
        tc.time = Instant::now();
        eprintln!("found existing key: {port}");
    }

    let stream = loop {
        match TcpStream::connect(&addr).await {
            Ok(s) => break s,
            Err(e) => {
                eprintln!("waiting for server to start: {e}");
                tokio::time::sleep(tokio::time::Duration::from_millis(400))
                    .await;
            }
        }
    };

    cleanup();

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
    if let Err(c) = cookie {
        resp.headers_mut().insert(SET_COOKIE, c.into());
    }

    Ok(resp.map(|b| b.boxed()))
}

fn start_cmd(
    cmd: Option<PType>,
    port: usize,
) -> Option<Result<Response<BoxBody<Bytes, hyper::Error>>, hyper::Error>> {
    if TABLE.read().unwrap().len() > 5 {
        eprintln!("too many open connections!");
        return Some(Ok(Response::default()));
    }
    let cmd = cmd.unwrap();
    let child = Command::new("python")
        .arg(cmd.as_str())
        .arg("--port")
        .arg(format!("{port}"))
        .process_group(0) // give subprocesses the same PGID
        .spawn()
        .unwrap();
    eprintln!("starting new instance on {port}");
    TABLE.write().unwrap().insert(
        port,
        TimedChild {
            time: std::time::Instant::now(),
            child,
            typ: cmd,
        },
    );
    // if cmd is None, it should already be in the table
    None
}

/// returns Ok(cookie) if it was loaded and Err(cookie) if not
fn check_cookie(req: &Request<hyper::body::Incoming>) -> Result<usize, usize> {
    match req.headers().get(COOKIE) {
        Some(s) => Ok(String::from_utf8_lossy(s.as_bytes())
            .parse::<usize>()
            .unwrap()),
        None => Err(COUNTER.fetch_add(1, Ordering::Relaxed)),
    }
}
