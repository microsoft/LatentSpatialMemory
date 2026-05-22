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

const heroVideos = comparisonScenes.map((scene) => `./assets/videos/${scene}/ours.mp4`);
const shuffle = (items) => {
  const copy = [...items];
  for (let index = copy.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [copy[index], copy[swapIndex]] = [copy[swapIndex], copy[index]];
  }
  return copy;
};

const heroReel = document.querySelector(".hero-reel");
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

  const loadVideo = (video, src) => {
    video.src = src;
    video.currentTime = 0;
    video.load();
    video.play().catch(() => {});
  };

  loadVideo(reelVideos[0], nextSource());
  loadVideo(reelVideos[1], nextSource());

  setInterval(() => {
    const nextIndex = activeIndex === 0 ? 1 : 0;
    loadVideo(reelVideos[nextIndex], nextSource());
    reelVideos[nextIndex].classList.add("is-active");
    reelVideos[activeIndex].classList.remove("is-active");
    activeIndex = nextIndex;
  }, 6500);
}

const comparisonMethods = ["ours", "spatia", "voyager", "gen3c", "vmem"];
const comparisonList = document.querySelector("#comparisonList");

const syncPlayRow = (row) => {
  const videos = [...row.querySelectorAll("video")];
  if (!videos.length) return;

  const ready = videos.every((video) => video.readyState >= 2);
  if (!ready) {
    if (row.dataset.syncing === "true") return;
    row.dataset.syncing = "true";
    videos.forEach((video) => {
      video.addEventListener(
        "loadeddata",
        () => {
          row.dataset.syncing = "false";
          if (row.dataset.inView === "true") syncPlayRow(row);
        },
        { once: true },
      );
    });
    return;
  }

  row.dataset.syncing = "false";
  if (row.dataset.started !== "true") {
    videos.forEach((video) => {
      video.currentTime = 0;
    });
    row.dataset.started = "true";
  }

  requestAnimationFrame(() => {
    videos.forEach((video) => {
      video.play().catch(() => {});
    });
  });
};

const syncVisibleRows = () => {
  document.querySelectorAll(".comparison-row").forEach((row) => {
    const rect = row.getBoundingClientRect();
    const inView = rect.bottom > -240 && rect.top < window.innerHeight + 240;
    if (inView) {
      row.dataset.inView = "true";
      syncPlayRow(row);
    }
  });
};

const rowObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      const row = entry.target;
      const videos = [...row.querySelectorAll("video")];
      if (entry.isIntersecting) {
        row.dataset.inView = "true";
        syncPlayRow(row);
      } else {
        row.dataset.inView = "false";
        row.dataset.started = "false";
        videos.forEach((video) => video.pause());
      }
    });
  },
  { rootMargin: "240px 0px" },
);

if (comparisonList) {
  comparisonScenes.forEach((scene) => {
    const row = document.createElement("section");
    row.className = "comparison-row";

    comparisonMethods.forEach((method) => {
      const cell = document.createElement("div");
      cell.className = "comparison-cell";

      const video = document.createElement("video");
      video.src = `./assets/videos/${scene}/${method}.mp4`;
      video.preload = "metadata";
      video.muted = true;
      video.loop = true;
      video.playsInline = true;
      video.addEventListener("loadeddata", syncVisibleRows, { once: true });

      cell.append(video);
      row.append(cell);
    });

    comparisonList.append(row);
    rowObserver.observe(row);
  });

  window.addEventListener("scroll", syncVisibleRows, { passive: true });
  window.addEventListener("hashchange", syncVisibleRows);
  requestAnimationFrame(syncVisibleRows);
}
