const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) entry.target.classList.add("is-visible");
    });
  },
  { threshold: 0.14 },
);

document.querySelectorAll(".reveal").forEach((element, index) => {
  element.style.transitionDelay = `${Math.min(index % 3, 2) * 70}ms`;
  revealObserver.observe(element);
});

const comparisonScenes = [
  "00000000",
  "00000001",
  "00000002",
  "00000003",
  "00011769",
  "00011843",
  "00011859",
  "00011994",
  "00011996",
  "00012066",
  "00012075",
  "00012212",
  "00012223",
  "00012259",
  "00012293",
];

const comparisonMethods = ["ours", "spatia", "voyager", "gen3c", "vmem"];
const heroVideos = comparisonScenes.map((scene) => `./assets/videos/${scene}/ours.mp4`);

const shuffle = (items) => {
  const copy = [...items];
  for (let index = copy.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [copy[index], copy[swapIndex]] = [copy[swapIndex], copy[index]];
  }
  return copy;
};

const playVideo = (video) => {
  video.play().catch(() => {});
};

const heroReel = document.querySelector(".hero-reel");
let heroReady = Promise.resolve();

if (heroReel) {
  const reelVideos = [...heroReel.querySelectorAll("video")];
  let playlist = shuffle(heroVideos);
  let playlistIndex = 0;
  let activeIndex = 0;

  const nextSource = () => {
    if (playlistIndex >= playlist.length) {
      playlist = shuffle(heroVideos);
      playlistIndex = 0;
    }

    const src = playlist[playlistIndex];
    playlistIndex += 1;
    return src;
  };

  const loadHeroVideo = (video, src) => {
    video.preload = "auto";
    video.src = src;
    video.currentTime = 0;
    video.load();
    playVideo(video);
  };

  heroReady = new Promise((resolve) => {
    const firstVideo = reelVideos[0];
    let resolved = false;

    const finish = () => {
      if (resolved) return;
      resolved = true;
      resolve();
    };

    firstVideo.addEventListener("canplay", finish, { once: true });
    firstVideo.addEventListener("loadeddata", finish, { once: true });
    setTimeout(finish, 2500);
  });

  loadHeroVideo(reelVideos[0], nextSource());
  loadHeroVideo(reelVideos[1], nextSource());

  setInterval(() => {
    const nextIndex = activeIndex === 0 ? 1 : 0;
    loadHeroVideo(reelVideos[nextIndex], nextSource());
    reelVideos[nextIndex].classList.add("is-active");
    reelVideos[activeIndex].classList.remove("is-active");
    activeIndex = nextIndex;
  }, 6500);
}

const comparisonList = document.querySelector("#comparisonList");
const rowLoadQueue = [];
let loadingRow = false;

const loadVideoSource = (video) => {
  if (video.dataset.loaded === "true") return;

  video.src = video.dataset.src;
  video.preload = "auto";
  video.dataset.loaded = "true";
  video.load();
};

const waitForRowReady = (row) => {
  const videos = [...row.querySelectorAll("video")];

  return Promise.race([
    Promise.all(
      videos.map(
        (video) =>
          new Promise((resolve) => {
            if (video.readyState >= 2) {
              resolve();
              return;
            }

            video.addEventListener("loadeddata", resolve, { once: true });
            video.addEventListener("error", resolve, { once: true });
          }),
      ),
    ),
    new Promise((resolve) => setTimeout(resolve, 5000)),
  ]);
};

const syncPlayRow = (row) => {
  const videos = [...row.querySelectorAll("video")];
  if (!videos.length) return;

  if (row.dataset.started !== "true") {
    videos.forEach((video) => {
      if (video.readyState >= 1) video.currentTime = 0;
    });
    row.dataset.started = "true";
  }

  requestAnimationFrame(() => {
    videos.forEach(playVideo);
  });
};

const processRowQueue = async () => {
  if (loadingRow) return;
  loadingRow = true;

  while (rowLoadQueue.length) {
    const row = rowLoadQueue.shift();
    if (!row || row.dataset.loaded === "true") continue;

    row.dataset.loading = "true";
    row.querySelectorAll("video").forEach(loadVideoSource);
    await waitForRowReady(row);

    row.dataset.loaded = "true";
    row.dataset.loading = "false";

    if (row.dataset.inView === "true") {
      syncPlayRow(row);
    }
  }

  loadingRow = false;
};

const requestRowLoad = (row) => {
  if (row.dataset.loaded === "true") {
    syncPlayRow(row);
    return;
  }

  if (row.dataset.queued === "true") return;

  row.dataset.queued = "true";
  rowLoadQueue.push(row);
  processRowQueue();
};

const rowObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      const row = entry.target;
      const videos = [...row.querySelectorAll("video")];

      if (entry.isIntersecting) {
        row.dataset.inView = "true";
        requestRowLoad(row);
      } else {
        row.dataset.inView = "false";
        row.dataset.started = "false";
        videos.forEach((video) => video.pause());
      }
    });
  },
  { rootMargin: "360px 0px" },
);

const buildComparisonRows = () => {
  if (!comparisonList || comparisonList.dataset.built === "true") return;

  comparisonList.dataset.built = "true";

  comparisonScenes.forEach((scene) => {
    const row = document.createElement("section");
    row.className = "comparison-row";

    comparisonMethods.forEach((method) => {
      const cell = document.createElement("div");
      cell.className = "comparison-cell";

      const video = document.createElement("video");
      video.dataset.src = `./assets/videos/${scene}/${method}.mp4`;
      video.preload = "none";
      video.muted = true;
      video.loop = true;
      video.playsInline = true;

      cell.append(video);
      row.append(cell);
    });

    comparisonList.append(row);
    rowObserver.observe(row);
  });
};

heroReady.then(() => {
  buildComparisonRows();
});
